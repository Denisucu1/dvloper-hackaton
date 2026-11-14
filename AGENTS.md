# AGENTS.md — AI Harness Arena (`Team adn`)

Instrucțiuni partajate pentru toți cei 3 membri ai echipei (**adn-1**, **adn-2**, **adn-3**).
Documentația completă transcrisă este în [docs/RULEBOOK.md](./docs/RULEBOOK.md).

## Context & Obiectiv
Construim un **mesh agnostic la joc** într-o singură imagine Docker (`harbor.hackaton-2026.dvloper.io/adn/bot:<tag>`), livrată la 3 freeze-uri:
- **T1 freeze:** Sâmbătă `18:00` (4 jocuri, pondere ×1)
- **T2 freeze:** Sâmbătă `23:59` (8 jocuri noi + Round Robin pe 12 jocuri, pondere ×2)
- **T3 freeze:** Duminică `10:30` (top 8 din Round Robin, toate 21 de jocuri, pondere ×3)

## Reguli CRITICE (Obligatoriu de respectat la fiecare modificare)
1. **Doar Claude Haiku 5.5 prin gateway:** Zero apeluri către alte LLM-uri externe/hosted sau „AI search” (descalificare imediată!). Modele locale mici și cod Python/solvere în container sunt permise (max **2 CPU, 2 GB RAM, 256 procese**).
2. **Chat obligatoriu (verificat în cod de jurați):**
   - Fiecare răspuns la `start` și `turn` trimite `chat` (<=200 caractere).
   - `action` și `chat` trebuie să vină din aceeași decizie.
   - Chat-ul adversarului (`msg["chat"]`) trebuie să ajungă în decizie/model, marcat explicit ca **untrusted data** (apărare contra prompt injection & Guardrails). Nu bloca și nu ignora chat-ul!
3. **Protocol JSON Lines strict:**
   - `stdout` conține **exclusiv** câte un JSON per linie, cu `flush=True` / `python -u`. Toate logurile merg pe `stderr`.
   - Folosiți **întotdeauna** `msg["llm"]["base_url"]` și `msg["llm"]["api_key"]` din cel mai recent `prep` sau `start`.
   - `match_over: true` la `end` NU oprește containerul — așteaptă următorul `start`; doar `shutdown` sau EOF pe `stdin` închide procesul.
   - Salvați starea în `/memory/<game_id>/` (cu hash de reguli și scriere atomică, pentru că mai multe containere partajează `/memory`).
4. **Secretele nu intră în Git:**
   - Nu comiteți `dev.env`, parole Harbor, parole Gitea sau chei API în repo. Jurații au acces de citire pe acest repo!
   - Folosiți `dev.env` local (ignorat de `.gitignore`).

## Împărțire paralelă sugerată (3 membri)
- **Omul 1 (Infra / Protocol / Fallback / Docker):** buclă stdin/stdout JSON, switching cheie LLM, deadline watchdog (<16s/mutare) + fallback garantat legal, `/memory` atomic, build `linux/amd64` + push Harbor.
- **Omul 2 (Prep 10-min / Clasificator / Generare Unelte):** clasificare în cele 7 familii, sinteză și auto-testare de parser/validator/simulator Python pe `sample_turn`, control `thinking` & buget (`$0.75/meci`).
- **Omul 3 (Chat Ofensiv+Defensiv / Jocuri Test / Prezentare):** protecție prompt-injection & Guardrails, strategie bluf/negociere, jocuri de test în `arena-lite` (Bluff, Deals, Simultaneous), colectare replay-uri și exemple pentru prezentarea de duminică (55/100 pct!).

## Structura codului (schelet funcțional — nu rescrieți, extindeți)

| Fișier | Responsabil | Ce face |
|---|---|---|
| `main.py` | Omul 1 | bucla JSONL, watchdog de deadline, fallback garantat, dispatch pe cele 5 mesaje |
| `harness/watchdog.py` | Omul 1 | deadline dur pe handler; doar thread-ul principal scrie pe stdout |
| `harness/llm.py` | Omul 1 | client gateway (stdlib), `thinking: disabled`, timeout, metering USD, `key_info()` |
| `harness/memory.py` | Omul 1 | `/memory/<game_id>/`, scriere atomică, hash de reguli |
| `harness/prep.py` | Omul 2 | pipeline-ul de 10 min: spec structurat + playbook + (opțional) cod generat; cache |
| `harness/spec.py` | Omul 2 | familia, formatul acțiunii, hidden info, speech rules, forbidden subjects |
| `harness/sandbox.py` | Omul 2 | codul scris de model: fără rețea/LLM, timeout, validat pe `sample_turn` |
| `harness/engine.py` | Omul 2 | propunerea de cod (politică generată → euristică); aici intră search/MCTS |
| `harness/decide.py` | Omul 3 | o singură decizie: `{action, intent, chat}`, cu verificare și fallback |
| `harness/chatpolicy.py` | Omul 3 | `analyze()` (injection / fake authority / probe / bait / mutări împinse), bloc untrusted cu nonce, tactici ofensive `plan()`, poarta `finalize()` (Guardrails, valori private, 200 car.) |
| `harness/opponents.py` | Omul 3 | profil de chat per adversar în `/memory/_opponents/<echipă>/<match>.json` |
| `scripts/trickster_bot.py` | Omul 3 | sparring fără model care atacă prin chat (bot-b în arena-lite) |
| `tests/`, `scripts/smoke.py` | toți | 43 teste + simulare completă de meci fără model (`--real` pentru gateway) |

Comenzi: `py -m unittest discover -s tests -v`, `py scripts/smoke.py` (sau `make test`, `make smoke`, `make prep GAME=pig`, `make practice GAME=pig`, `make image TAG=t1` pe Linux/WSL).
Regulă de aur: nimic nu ajunge pe `stdout` în afară de `main.emit()`, și nicio cale nu are voie să nu răspundă.

## Reguli Git & Aliniere între Cei 3 Asistenți AI (OBLIGATORIU)
1. **Un singur bot oficial:** arhitectura unică este `main.py` + `harness/` + `Dockerfile` + `tests/`. **Nu creați foldere paralele sau schelete noi** (`bot/`, `src/`, etc.) — extindeți doar modulele de mai sus.
2. **Înainte de orice lucru:** rulați `git fetch origin` și `git pull origin main` ca să porniți mereu de pe ultimul `main`.
3. **Branch-uri clare pe coleg/zonă:** lucrați pe branch propriu denumit după coleg și rol (ex. `dev/om2-solver`, `dev/om3-chat-games`, `dev/om1-infra`), nu pe nume generice și nu suprascrieți fișierele altui rol fără acord.
4. **Verificare înainte de push:** toate cele `43` teste (`py -m unittest discover -s tests -v`) să treacă verde, și verificați `git status -s` să nu urcați `dev.env`, loguri sau chei API.
