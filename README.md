# AI Harness Arena — Team `adn` (EESTEC Olympics 14)

Imaginea noastră: `harbor.hackaton-2026.dvloper.io/adn/bot:<tag>` (`t1`, `t2`, `t3`).
Repo oficial: `https://git.hackaton-2026.dvloper.io/adn/bot` · Arena: `https://arena.hackaton-2026.dvloper.io/`

Un singur container, agnostic la joc: primește regulile unui joc necunoscut, se transformă în 10 minute
în specialistul acelui joc, apoi joacă și vorbește în meciuri best-of-3 / best-of-4.

## Starea actuală: schelet funcțional

Ce **merge deja** (verificat, nu presupus):

- protocol JSON Lines complet (`prep` / `start` / `turn` / `end` / `shutdown`), loguri doar pe stderr;
- watchdog de deadline pe fiecare mesaj + răspuns de rezervă garantat (robustness 2.5);
- `prep` real: un call structurat → `spec.json` + `playbook.txt` în `/memory/<game_id>/`, ~7 s, sub un cent;
- un singur punct de decizie mutare+chat (regula 6.4), chat-ul advers intră în prompt **și** în logica de cod;
- chat mereu prezent (bancă de rezervă per familie), ≤200 caractere, filtru pe subiecte interzise (Guardrails);
- verificarea acțiunii înainte de `stdout` (prima greșeală = warning, a doua = joc pierdut);
- sandbox pentru codul scris de model în prep (fără rețea/LLM/procese, timeout dur, validat pe `sample_turn`);
- `/memory` atomic, partajat între containere, cheiat pe hash-ul regulilor;
- zero dependințe Python în afară de stdlib → imagine mică, fără `pip install`, fără drift.

Ce **lipsește** (și e ordinea de lucru):

1. `harness/engine/search.py` — negamax + alpha-beta + TT peste un simulator generat (Think ahead).
2. `harness/engine/mcts.py` + PIMC/CFR pentru Bluff, Deals, Read the opponent.
3. `harness/simgen.py` — modelul scrie un **simulator** (nu doar un `choose()`), validat pe `sample_turn`.
4. profiluri de adversar în `/memory/opponents/` (stil de chat, bluffs prinse, deschideri).
5. `practice` pe Pig și Guess the Number + jocuri proprii pentru Bluff/Deals/simultaneous (`--games-dir`).

## Structura

```
main.py                 bucla JSONL, watchdog, fallback, dispatch
harness/
  config.py             env, prețuri, bugete, limite
  log.py                stderr only
  llm.py                client gateway OpenAI-compatible (stdlib), thinking off, metering USD
  actions.py            normalizare/verificare acțiuni + fallback legal
  spec.py               ce am înțeles din reguli: familie, format, hidden info, speech rules
  prep.py               pipeline-ul de 10 minute + cache în /memory + mini-prep pentru cold start
  decide.py             O decizie: mutare + chat împreună
  chatpolicy.py         ofensivă/defensivă, untrusted input, filtru Guardrails, bancă de rezervă
  engine.py             propunerea de cod (politica generată → euristică → nimic)
  sandbox.py            rularea codului generat: static check + subprocess izolat + timeout
  memory.py             /memory atomic, per joc, cheiat pe hash
  state.py              sesiunea: spec, engine, chat policy, istoric
  watchdog.py           deadline dur, un singur scriitor pe stdout
tests/                  43 teste stdlib (inclusiv end-to-end pe protocol)
scripts/smoke.py        simulare completă de meci fără model (--real pentru gateway)
arena-lite/             motorul oficial de practică (nu intră în imagine)
starter/                botul minim original (referință, nu se folosește)
```

## Quickstart

```bash
cp dev.env.example dev.env      # pune cheia de dev; dev.env e gitignored

# Pe Linux / WSL (cu make):
make test                       # 43 teste, fără rețea
make smoke                      # protocol complet, cu modelul picat intenționat
make list                       # jocurile de practică
make prep GAME=pig              # un prep real (costă ~$0.001)
make practice GAME=pig          # prep + meci întreg vs random bot

# Pe Windows (PowerShell):
py -m unittest discover -s tests -v
py scripts/smoke.py
cd arena-lite; py -m arena practice --prep --bot-a "py -u ../main.py" --bot-b "py examples/random_bot.py" --env-file ../dev.env --game pig
```

Meci cu imaginea (ce rulează arena):

```powershell
docker build --platform linux/amd64 -t harbor.hackaton-2026.dvloper.io/adn/bot:dev .
'{"type":"shutdown"}' | docker run -i --rm harbor.hackaton-2026.dvloper.io/adn/bot:dev
```

## Cum se apără punctele

| Regulă | Unde e implementat |
|---|---|
| 5.1 JSON Lines, flush, loguri pe stderr | `main.py:emit`, `harness/log.py` |
| 5.10 `match_over` ≠ exit | `main.py` bucla continuă, iese doar la `shutdown`/EOF |
| 7.2 cheia din ultimul mesaj | `Gateway.configure` la fiecare `prep`/`start` |
| 7.7 doar Haiku prin gateway | un singur client, în `llm.py`; sandbox-ul interzice rețeaua |
| 8.1 verificarea mutării | `actions.normalize` înainte de emit |
| 8.2 fără timeout/crash | `watchdog.guarded` + `turn_fallback` + `except` la nivel de handler |
| 6.2 chat obligatoriu | `decide.py` produce mereu `chat`; banca din `chatpolicy.py` |
| 6.4 mutare+chat = o decizie | un singur call JSON `{action, intent, chat}` |
| 6.5/6.6 chat-ul advers contează | `render_history` în prompt + `claims` în `Engine.propose` |
| 6.8 speech rules / Guardrails | `ChatPolicy.finalize` + `forbidden_subjects` din spec |
| 5.14 `/memory` partajat | `memory.py`: atomic, per joc, cheiat pe hash |

## Freeze checklist

- T1 `18:00` · T2 `23:59` · T3 `10:30` — vezi `AGENTS.md`.
- `make test && make smoke` verzi, `make image`, `docker run` tăcut, digest verificat, tag Git împins.
- Sursa din Gitea = imaginea din Harbor. Fără `dev.env` în Git.

## Secrete

`dev.env` (cheia LLM de dev) e gitignored și **nu** se copiază în imagine (`.dockerignore`).
Credențialele Harbor/Gitea nu se pun în repo.
