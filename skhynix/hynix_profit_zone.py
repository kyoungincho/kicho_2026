"""SK하이닉스 보유 포지션 향후 수입(매도) 구간 분석기.

국내 상장주식(코스피) 개인(소액주주) 기준으로 매도 시 실수령액과 손익을
매도 목표가별로 계산한다. 세금/수수료 가정은 CONFIG 에서 조정할 수 있다.

주의: 본 스크립트는 투자 판단 보조용 계산기일 뿐이며 투자 자문/권유가 아니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Position:
    """보유 포지션 정보."""

    shares: int = 42                 # 보유 수량 (주)
    avg_price: float = 2_264_880.0   # 평균단가 (평단, 원)
    current_price: float = 2_425_000.0  # 기준 현재가 (2026-07-03 종가, 원)

    @property
    def cost(self) -> float:
        """총 매입금액 (원)."""
        return self.shares * self.avg_price

    @property
    def market_value(self) -> float:
        """현재 평가금액 (원)."""
        return self.shares * self.current_price


@dataclass
class FeeModel:
    """매도 시 비용 모델 (코스피, 개인 소액주주 기준)."""

    # 증권거래세: 2026년 코스피 0.15% (농특세 제외한 실효 세율 가정)
    transaction_tax_rate: float = 0.0015
    # 위탁매매 수수료: 증권사별 상이, 유관기관 제비용 포함 대략치
    brokerage_fee_rate: float = 0.00015
    # 국내 상장주식 소액주주 양도소득세 없음 (대주주 아님 가정)
    capital_gains_tax_rate: float = 0.0

    def sell_costs(self, gross: float, gain: float) -> float:
        """매도금액(gross)과 차익(gain)에 대한 총 비용."""
        tax = gross * self.transaction_tax_rate
        fee = gross * self.brokerage_fee_rate
        cgt = max(gain, 0.0) * self.capital_gains_tax_rate
        return tax + fee + cgt


@dataclass
class Scenario:
    label: str
    sell_price: float
    note: str = ""


@dataclass
class Result:
    scenario: Scenario
    gross_proceeds: float
    costs: float
    net_proceeds: float
    net_profit: float
    return_pct: float
    vs_current_pct: float


def analyze(position: Position, fees: FeeModel, scenarios: list[Scenario]) -> list[Result]:
    results: list[Result] = []
    for sc in scenarios:
        gross = position.shares * sc.sell_price
        gross_gain = gross - position.cost
        costs = fees.sell_costs(gross, gross_gain)
        net = gross - costs
        net_profit = net - position.cost
        return_pct = net_profit / position.cost * 100
        vs_current = (sc.sell_price / position.current_price - 1) * 100
        results.append(
            Result(
                scenario=sc,
                gross_proceeds=gross,
                costs=costs,
                net_proceeds=net,
                net_profit=net_profit,
                return_pct=return_pct,
                vs_current_pct=vs_current,
            )
        )
    return results


def default_scenarios(position: Position) -> list[Scenario]:
    """평단 대비 손익 구간 + 애널리스트 목표가 구간 시나리오."""
    avg = position.avg_price
    return [
        Scenario("손절 -10%", avg * 0.90, "리스크 관리 하단"),
        Scenario("본전 (평단)", avg, "손익분기 근처"),
        Scenario("현재가", position.current_price, "2026-07-03 종가"),
        Scenario("1차 익절 +10%", avg * 1.10, "단기 분할 매도 후보"),
        Scenario("2차 익절 +20%", avg * 1.20, "컨센서스 하단 근처"),
        Scenario("컨센서스 평균", 3_100_000, "애널리스트 평균 목표가"),
        Scenario("컨센서스 상단", 3_320_000, "일부 컨센서스 상단"),
        Scenario("공격적 목표 +50%", avg * 1.50, "강세 시나리오"),
        Scenario("초강세 목표", 4_700_000, "최고 목표가(노무라 등)"),
    ]


def _won(x: float) -> str:
    return f"{x:,.0f}원"


def render_table(results: list[Result], position: Position) -> str:
    lines: list[str] = []
    header = (
        "| 시나리오 | 매도가 | 현재가대비 | 실수령액 | 순손익 | 수익률 | 비고 |",
        "|---|---:|---:|---:|---:|---:|---|",
    )
    lines.extend(header)
    for r in results:
        sc = r.scenario
        lines.append(
            "| {label} | {sell} | {vs:+.1f}% | {net} | {profit} | {ret:+.1f}% | {note} |".format(
                label=sc.label,
                sell=_won(sc.sell_price),
                vs=r.vs_current_pct,
                net=_won(r.net_proceeds),
                profit=_won(r.net_profit),
                ret=r.return_pct,
                note=sc.note,
            )
        )
    return "\n".join(lines)


def summary(position: Position, fees: FeeModel) -> str:
    unrealized = position.market_value - position.cost
    unrealized_pct = unrealized / position.cost * 100
    return (
        f"- 보유수량: {position.shares:,}주\n"
        f"- 평단(평균단가): {_won(position.avg_price)}\n"
        f"- 총 매입금액: {_won(position.cost)}\n"
        f"- 현재가(2026-07-03): {_won(position.current_price)}\n"
        f"- 평가금액: {_won(position.market_value)}\n"
        f"- 평가손익(세전): {_won(unrealized)} ({unrealized_pct:+.2f}%)\n"
        f"- 세율 가정: 증권거래세 {fees.transaction_tax_rate*100:.2f}%, "
        f"수수료 {fees.brokerage_fee_rate*100:.3f}%, "
        f"양도세 {fees.capital_gains_tax_rate*100:.1f}%(소액주주 비과세)"
    )


def main() -> None:
    position = Position()
    fees = FeeModel()
    scenarios = default_scenarios(position)
    results = analyze(position, fees, scenarios)

    print("=== SK하이닉스 보유 포지션 요약 ===")
    print(summary(position, fees))
    print()
    print("=== 향후 수입(매도) 구간 분석 ===")
    print(render_table(results, position))
    print()
    print("※ 본 결과는 계산 보조용이며 투자 자문/권유가 아닙니다.")


if __name__ == "__main__":
    main()
