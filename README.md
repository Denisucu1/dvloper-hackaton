# AI Harness Arena — Team Bot (`EESTEC Olympics 14`)

Repository pentru botul / mesh-ul echipei la provocarea **AI Harness Arena** (`dvloper.io`).

## Structură planificată (după primirea starter kit-ului)

- `bot/` (sau `starter/`) — codul harness-ului, `Dockerfile` (`linux/amd64`) și `/logo.png` opțional
- `arena-lite/` — simulatorul local primit de la organizatori (pentru testare locală)
- `dev.env` — credențiale locale de dev (**neversionat**, ignorat în `.gitignore`, vezi `dev.env.example`)

## Comenzi esențiale

### 1. Testare locală cu `arena-lite` (din folderul `arena-lite`)
```powershell
py -m arena practice --list
py -m arena practice --bot-a "py ../starter/main.py" --bot-b "python examples/random_bot.py" --env-file ../dev.env --game tic_tac_toe
```

### 2. Build & push imagine Docker (`linux/amd64`)
```powershell
docker build --platform linux/amd64 -t <registry>/<team>/bot:t1 .
docker login <registry> -u '<robot user>' -p '<secret>'
docker push <registry>/<team>/bot:t1
docker buildx imagetools inspect <registry>/<team>/bot:t1
```

## Checklist obligatoriu la fiecare freeze (`t1`, `t2`, `t3`)

- [ ] **T1 freeze:** Sâmbătă `18:00` | **T2 freeze:** Sâmbătă `23:59` | **T3 freeze:** Duminică `10:30`
- [ ] Acces **read** acordat conturilor juraților pe repo-ul Git, cu commit/tag corespunzător fiecărui freeze (`t1`, `t2`, `t3`)
- [ ] Imagine construită pentru `linux/amd64` și verificată cu `docker buildx imagetools inspect`
- [ ] Zero chei secrete în Git sau în imagine (`dev.env` rămâne exclus)
- [ ] Respectă protocolul JSON Lines (`stdout` unbuffered, logs pe `stderr`), regula Haiku-only și regula de chat obligatoriu
