# kicho_2026

`대한민국 32강 진출 시나리오 계산기` — a 2026 FIFA World Cup Korea Round-of-32 qualification calculator.

## Cursor Cloud specific instructions

- This is a **dependency-free static web app** (`index.html`, `styles.css`, `app.js` — vanilla HTML/CSS/JS). There is no package manager, lockfile, build step, linter, or test framework. Nothing needs to be installed; the update script is a no-op.
- Run it by serving the directory that contains `index.html` and opening it in a browser. From that directory: `python3 -m http.server 8000`, then open `http://localhost:8000/index.html` (Python 3 and Chrome are preinstalled). Opening `index.html` via `file://` also works.
- All logic runs client-side in `app.js`; there is no backend, database, or external API. To manually test: enter the 6 match scores, click `결과 계산하기`, and confirm the standings table + qualification banner render.
- Note: the product files may live on a feature branch rather than `main` (on which only `README.md`/`AGENTS.md` may be present). If `index.html` is missing from the working tree, check out the branch that contains the app before serving it.
