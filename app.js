// 2026 월드컵 대한민국 32강 진출 계산기
// 4팀 라운드로빈 = 6경기. 대한민국은 인덱스 0.

const KOREA = 0;

// 4팀 라운드로빈 대진 (home, away)
const FIXTURES = [
  [0, 1], [2, 3],
  [0, 2], [1, 3],
  [0, 3], [1, 2],
];

function teamName(i) {
  const el = document.getElementById("team" + i);
  return (el && el.value.trim()) || "팀 " + (i + 1);
}

function renderMatches() {
  const wrap = document.getElementById("matches");
  wrap.innerHTML = "";
  FIXTURES.forEach((fx, idx) => {
    const [h, a] = fx;
    const row = document.createElement("div");
    row.className = "match";
    row.innerHTML = `
      <span class="name home" data-team="${h}">${teamName(h)}</span>
      <input type="number" min="0" max="99" id="h${idx}" placeholder="-" />
      <span class="vs">vs</span>
      <input type="number" min="0" max="99" id="a${idx}" placeholder="-" />
      <span class="name away" data-team="${a}">${teamName(a)}</span>
    `;
    wrap.appendChild(row);
  });
}

// 팀 이름 변경 시 대진표 이름도 갱신
function syncNames() {
  document.querySelectorAll(".name[data-team]").forEach((el) => {
    el.textContent = teamName(parseInt(el.dataset.team, 10));
  });
}

function computeStandings() {
  const stats = [0, 1, 2, 3].map((i) => ({
    i,
    name: teamName(i),
    P: 0, W: 0, D: 0, L: 0, GF: 0, GA: 0, GD: 0, Pts: 0,
  }));

  let allFilled = true;

  FIXTURES.forEach((fx, idx) => {
    const [h, a] = fx;
    const hv = document.getElementById("h" + idx).value;
    const av = document.getElementById("a" + idx).value;
    if (hv === "" || av === "") { allFilled = false; return; }
    const hs = parseInt(hv, 10);
    const as = parseInt(av, 10);
    if (Number.isNaN(hs) || Number.isNaN(as)) { allFilled = false; return; }

    stats[h].P++; stats[a].P++;
    stats[h].GF += hs; stats[h].GA += as;
    stats[a].GF += as; stats[a].GA += hs;

    if (hs > as) { stats[h].W++; stats[a].L++; stats[h].Pts += 3; }
    else if (hs < as) { stats[a].W++; stats[h].L++; stats[a].Pts += 3; }
    else { stats[h].D++; stats[a].D++; stats[h].Pts += 1; stats[a].Pts += 1; }
  });

  stats.forEach((s) => (s.GD = s.GF - s.GA));

  // 정렬: 승점 → 골득실 → 득점 → 이름
  const sorted = [...stats].sort(
    (x, y) =>
      y.Pts - x.Pts || y.GD - x.GD || y.GF - x.GF || x.name.localeCompare(y.name)
  );
  sorted.forEach((s, idx) => (s.pos = idx + 1));

  return { sorted, allFilled };
}

function buildTable(sorted) {
  let rows = sorted
    .map((s) => {
      const isKorea = s.i === KOREA;
      return `
      <tr class="${isKorea ? "korea" : ""}">
        <td><span class="pos-badge pos-${s.pos}">${s.pos}</span></td>
        <td class="team-name">${s.name}${isKorea ? " 🇰🇷" : ""}</td>
        <td>${s.P}</td><td>${s.W}</td><td>${s.D}</td><td>${s.L}</td>
        <td>${s.GF}</td><td>${s.GA}</td>
        <td>${s.GD > 0 ? "+" + s.GD : s.GD}</td>
        <td><strong>${s.Pts}</strong></td>
      </tr>`;
    })
    .join("");

  return `
    <table>
      <thead>
        <tr>
          <th>순위</th><th style="text-align:left">팀</th>
          <th>경기</th><th>승</th><th>무</th><th>패</th>
          <th>득</th><th>실</th><th>차</th><th>승점</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>`;
}

function calc() {
  syncNames();
  const { sorted, allFilled } = computeStandings();
  const korea = sorted.find((s) => s.i === KOREA);
  const result = document.getElementById("result");
  result.classList.remove("hidden");

  let banner = "";
  if (!allFilled) {
    banner = `<div class="banner maybe">⏳ 아직 입력되지 않은 경기가 있어 <strong>중간 집계</strong>입니다. 모든 스코어를 채우면 최종 순위가 확정됩니다.</div>`;
  } else if (korea.pos <= 2) {
    banner = `<div class="banner ok">✅ 대한민국 조 <strong>${korea.pos}위</strong> — <strong>32강 자동 진출!</strong> (승점 ${korea.Pts}, 골득실 ${korea.GD > 0 ? "+" + korea.GD : korea.GD})</div>`;
  } else if (korea.pos === 3) {
    banner = `<div class="banner maybe">🥉 대한민국 조 <strong>3위</strong> — 12개 조 3위 팀 중 <strong>상위 8위</strong>에 들면 진출 가능. 아래에서 비교해 보세요.</div>`;
  } else {
    banner = `<div class="banner out">❌ 대한민국 조 <strong>4위</strong> — 아쉽지만 32강 진출이 어렵습니다.</div>`;
  }

  result.innerHTML = banner + buildTable(sorted);

  // 3위면 3위 비교 섹션으로 부드럽게 안내
  if (allFilled && korea.pos === 3) {
    prefillKoreaThird(korea);
    document.getElementById("thirdSection").scrollIntoView({ behavior: "smooth" });
  }
}

// ---------- 3위 경쟁 비교 ----------
// 12개 조 중 우리 조 1팀(대한민국)을 제외한 나머지 11개 조의 3위 가정값과 비교.
// 사용자가 비워두면 통상적인 진출 커트라인(승점 3, 골득실 -1, 득점 2)을 11팀에 적용.

const OTHER_THIRDS = 11; // 대한민국 외 다른 조 3위 팀 수
const QUALIFY_SLOTS = 8; // 3위 중 진출 자리

function buildThirdInputs() {
  const wrap = document.getElementById("thirdPlaceInputs");
  let html = `
    <div class="third-row head">
      <span class="lbl">조 (3위 팀)</span>
      <span class="lbl">승점</span>
      <span class="lbl">골득실</span>
      <span class="lbl">득점</span>
    </div>
    <div class="third-row">
      <span class="lbl">🇰🇷 대한민국</span>
      <input type="number" id="kPts" placeholder="승점" />
      <input type="number" id="kGD" placeholder="골득실" />
      <input type="number" id="kGF" placeholder="득점" />
    </div>`;
  for (let g = 0; g < OTHER_THIRDS; g++) {
    html += `
    <div class="third-row">
      <span class="lbl">다른 조 ${g + 1}</span>
      <input type="number" class="oPts" placeholder="3 (기본)" />
      <input type="number" class="oGD" placeholder="-1 (기본)" />
      <input type="number" class="oGF" placeholder="2 (기본)" />
    </div>`;
  }
  wrap.innerHTML = html;
}

function prefillKoreaThird(korea) {
  document.getElementById("kPts").value = korea.Pts;
  document.getElementById("kGD").value = korea.GD;
  document.getElementById("kGF").value = korea.GF;
}

function num(v, def) {
  if (v === "" || v === null || v === undefined) return def;
  const n = parseInt(v, 10);
  return Number.isNaN(n) ? def : n;
}

function compareThirds() {
  const k = {
    name: "🇰🇷 대한민국",
    Pts: num(document.getElementById("kPts").value, 3),
    GD: num(document.getElementById("kGD").value, 0),
    GF: num(document.getElementById("kGF").value, 0),
    isKorea: true,
  };

  const others = [];
  const oPts = document.querySelectorAll(".oPts");
  const oGD = document.querySelectorAll(".oGD");
  const oGF = document.querySelectorAll(".oGF");
  for (let g = 0; g < OTHER_THIRDS; g++) {
    others.push({
      name: "다른 조 " + (g + 1),
      Pts: num(oPts[g].value, 3),
      GD: num(oGD[g].value, -1),
      GF: num(oGF[g].value, 2),
      isKorea: false,
    });
  }

  const all = [k, ...others].sort(
    (x, y) => y.Pts - x.Pts || y.GD - x.GD || y.GF - x.GF
  );
  const rank = all.findIndex((t) => t.isKorea) + 1;

  const out = document.getElementById("thirdResult");
  out.classList.remove("hidden");

  let banner;
  if (rank <= QUALIFY_SLOTS) {
    banner = `<div class="banner ok">✅ 3위 팀 12개 중 대한민국 <strong>${rank}위</strong> — 상위 8위 안! <strong>32강 진출 성공!</strong></div>`;
  } else {
    banner = `<div class="banner out">❌ 3위 팀 12개 중 대한민국 <strong>${rank}위</strong> — 8위 밖이라 진출이 어렵습니다. 승점·골득실을 더 쌓아야 합니다.</div>`;
  }

  const rows = all
    .slice(0, 12)
    .map(
      (t, i) => `
      <tr class="${t.isKorea ? "korea" : ""}">
        <td><span class="pos-badge ${i < QUALIFY_SLOTS ? "pos-1" : "pos-4"}">${i + 1}</span></td>
        <td class="team-name">${t.name}</td>
        <td>${t.Pts}</td>
        <td>${t.GD > 0 ? "+" + t.GD : t.GD}</td>
        <td>${t.GF}</td>
      </tr>`
    )
    .join("");

  out.innerHTML =
    banner +
    `<table><thead><tr><th>순위</th><th style="text-align:left">3위 팀</th><th>승점</th><th>골득실</th><th>득점</th></tr></thead><tbody>${rows}</tbody></table>
     <p class="hint">상위 8위(초록)까지 32강 진출, 9~12위(빨강)는 탈락입니다.</p>`;
}

// ---------- 초기화 ----------
document.addEventListener("DOMContentLoaded", () => {
  renderMatches();
  buildThirdInputs();
  [0, 1, 2, 3].forEach((i) => {
    document.getElementById("team" + i).addEventListener("input", syncNames);
  });
  document.getElementById("calcBtn").addEventListener("click", calc);
  document.getElementById("thirdBtn").addEventListener("click", compareThirds);
});
