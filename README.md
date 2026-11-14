# AI Harness Arena — Team `adn` (`EESTEC Olympics 14`)

Repository oficial Gitea: `https://git.hackaton-2026.dvloper.io/adn/bot`  
Imagine Docker Harbor: `harbor.hackaton-2026.dvloper.io/adn/bot:<tag>` (`t1`, `t2`, `t3`)  
Arena & Practice UI: `https://arena.hackaton-2026.dvloper.io/#/practice`

## Pentru colegii de echipă (`adn-1`, `adn-2`, `adn-3`)

1. Clonează repo-ul și creează local `dev.env` din `dev.env.example` (cu cheia `LLM_API_KEY` primită privat — **nu comite `dev.env`!**).
2. Citește [AGENTS.md](./AGENTS.md) și [docs/RULEBOOK.md](./docs/RULEBOOK.md).
3. Skill-urile din `.agents/skills/` sunt încărcate automat de asistenții AI compatibili.

## Comenzi esențiale

### 1. Testare locală cu `arena-lite` (după ce adăugăm starter kit-ul)
```powershell
py -m arena practice --list
py -m arena practice --bot-a "py ../starter/main.py" --bot-b "py examples/random_bot.py" --env-file ../dev.env --game tic_tac_toe
```

### 2. Build & push imagine Docker (`linux/amd64`)
```powershell
docker build --platform linux/amd64 -t harbor.hackaton-2026.dvloper.io/adn/bot:t1 .
docker login harbor.hackaton-2026.dvloper.io -u 'robot$adn+bot' -p '<secret>'
docker push harbor.hackaton-2026.dvloper.io/adn/bot:t1
docker buildx imagetools inspect harbor.hackaton-2026.dvloper.io/adn/bot:t1
```

> **Atenție:** Numele de robot Harbor (`robot$adn+bot`) conține `$` — păstrează ghilimelele simple `'robot$adn+bot'` în terminal (în PowerShell, ghilimelele simple previn interpretarea `$adn` ca variabilă!).

## Checklist obligatoriu la fiecare freeze (`t1`, `t2`, `t3`)

- [ ] **T1 freeze:** Sâmbătă `18:00` | **T2 freeze:** Sâmbătă `23:59` | **T3 freeze:** Duminică `10:30`
- [ ] Codul sursă și tag-ul Git (`t1`, `t2`, `t3`) împinse pe Gitea (`adn/bot`) să corespundă exact cu imaginea din Harbor
- [ ] Imagine construită pentru `linux/amd64` și verificată cu `docker buildx imagetools inspect`
- [ ] Zero chei secrete în Git sau în imagine (`dev.env` rămâne exclus)
- [ ] Respectă protocolul JSON Lines (`stdout` unbuffered, logs pe `stderr`), regula Haiku-only și regula de chat obligatoriu
