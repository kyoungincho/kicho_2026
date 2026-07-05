#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SK하이닉스(000660) 보유분 수익구간 분석기.

보유 수량과 평균단가(평단)를 입력하면, 다양한 매도가 시나리오별로
세후 순수익과 수익률을 계산해 '수입구간'을 표로 보여준다.

* 본 스크립트는 정보/계산 보조용이며 투자 자문이 아니다.
* 매도 비용은 KOSPI 기준 증권거래세 0.15% + 위탁수수료 약 0.015%를
  합쳐 0.165%로 근사한다(증권사/시점에 따라 다를 수 있음).
"""

from __future__ import annotations

from dataclasses import dataclass


# --- 기본 가정값 (필요시 수정) ---------------------------------------------
SHARES = 42                 # 보유 수량 (주)
AVG_PRICE = 2_264_880       # 평균단가 / 평단 (원)
SELL_COST_RATE = 0.00165    # 매도 비용률 (거래세 0.15% + 수수료 0.015%)

# 참고 시세 (2026-07-03 기준, 언론 보도)
CURRENT_PRICE = 2_425_000   # 현재가 근사
LOW_PRICE = 2_187_000       # 7/2 급락 저점

# 증권사 목표주가 (2026-07 초)
BROKER_TARGETS = {
    "상상인증권": 3_800_000,
    "IBK투자증권": 4_000_000,
    "교보증권": 4_000_000,
    "NH투자증권": 4_100_000,
    "KB증권": 4_200_000,
}


@dataclass
class ScenarioResult:
    label: str
    sell_price: int
    gross: float          # 총 매도금액 (수량 x 가격)
    sell_cost: float      # 매도 비용
    net_proceeds: float   # 세후 순 매도금액
    net_profit: float     # 순수익 (세후 - 총원가)
    return_pct: float     # 수익률 (%)


def analyze(sell_price: int, label: str,
            shares: int = SHARES, avg_price: int = AVG_PRICE) -> ScenarioResult:
    total_cost = shares * avg_price
    gross = shares * sell_price
    sell_cost = gross * SELL_COST_RATE
    net_proceeds = gross - sell_cost
    net_profit = net_proceeds - total_cost
    return_pct = net_profit / total_cost * 100
    return ScenarioResult(
        label=label,
        sell_price=sell_price,
        gross=gross,
        sell_cost=sell_cost,
        net_proceeds=net_proceeds,
        net_profit=net_profit,
        return_pct=return_pct,
    )


def won(n: float) -> str:
    return f"{n:,.0f}원"


def eok(n: float) -> str:
    """원 -> '억/만원' 가독 표기."""
    sign = "-" if n < 0 else ""
    n = abs(n)
    eok_part = int(n // 100_000_000)
    man_part = int((n % 100_000_000) // 10_000)
    if eok_part:
        return f"{sign}{eok_part}억 {man_part:,}만원"
    return f"{sign}{man_part:,}만원"


def build_scenarios() -> list[ScenarioResult]:
    total_cost = SHARES * AVG_PRICE
    scenarios: list[ScenarioResult] = []

    # 세후 손익분기(본전) 매도가: gross*(1-rate) = total_cost
    breakeven = total_cost / (SHARES * (1 - SELL_COST_RATE))
    scenarios.append(analyze(round(breakeven), "손익분기(세후 본전)"))

    scenarios.append(analyze(LOW_PRICE, "7/2 급락 저점"))
    scenarios.append(analyze(CURRENT_PRICE, "현재가(근사)"))

    for price in (2_600_000, 2_800_000, 3_000_000, 3_200_000, 3_500_000):
        scenarios.append(analyze(price, f"{price // 10_000:,}만원 구간"))

    for broker, target in sorted(BROKER_TARGETS.items(), key=lambda x: x[1]):
        scenarios.append(analyze(target, f"목표가 {target // 10_000:,}만 ({broker})"))

    return scenarios


def print_report() -> None:
    total_cost = SHARES * AVG_PRICE
    print("=" * 78)
    print("SK하이닉스(000660) 보유분 수익구간 분석")
    print("=" * 78)
    print(f"보유 수량 : {SHARES}주")
    print(f"평균단가  : {won(AVG_PRICE)}")
    print(f"총 매수원가: {won(total_cost)}  ({eok(total_cost)})")
    print(f"매도 비용률: {SELL_COST_RATE * 100:.3f}% (거래세+수수료 근사)")
    print("-" * 78)
    header = f"{'구간':<24}{'매도가':>13}{'세후순수익':>18}{'수익률':>10}"
    print(header)
    print("-" * 78)
    for s in build_scenarios():
        print(f"{s.label:<24}{won(s.sell_price):>13}"
              f"{eok(s.net_profit):>18}{s.return_pct:>9.1f}%")
    print("=" * 78)
    print("※ 정보 제공용 계산이며 투자 자문이 아님. 세율/수수료는 시점·증권사별 상이.")


if __name__ == "__main__":
    print_report()
