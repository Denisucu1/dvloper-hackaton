---
name: hackathon-deploy
description: >-
  Use before T1 (18:00), T2 (23:59), or T3 (10:30) freezes in AI Harness Arena
  to verify Docker build (linux/amd64), arena-lite practice tests, stdout/stderr
  cleanliness, Haiku-only & chat compliance, Harbor push, and Git judge access.
  Trigger in English or Romanian for deploy, freeze checklist, Harbor push, or
  pre-submission verification.
---

# AI Harness Arena — Pre-Freeze & Push Checklist

Run this verification before pushing `t1`, `t2`, or `t3` to Harbor. Any freeze miss or compliance breach costs the entire tournament or causes disqualification.

## 1. Automated & Manual Verification Gates (Rule 14.7)
- [ ] **Architecture:** Built with `docker build --platform linux/amd64 -t <registry>/<team>/bot:<tag> .`
- [ ] **Idle startup:** `docker run -i --rm <image>` starts with no args, waits silently on `stdin` without exiting or printing to `stdout`.
- [ ] **Clean `stdout`/`stderr`:** Only valid single-line JSON (`flush=True` / `python -u`) goes to `stdout`; all debug logs go to `stderr`.
- [ ] **Practice `--prep` passes:** Full match with `arena-lite` (`--prep`) finishes in time with **zero warnings and zero timeouts**.
- [ ] **Multi-match survival:** After `end` with `match_over: true`, container stays alive waiting for the next `start` (exits only on `shutdown` or closed `stdin`).
- [ ] **Dynamic key switching:** Uses `msg["llm"]["api_key"]` from the **latest** `prep` or `start` message, not only startup `LLM_API_KEY`.
- [ ] **Timeout/Empty fallback:** With Haiku slow, `thinking` eating tokens, or budget exhausted, bot still emits a valid fallback move within the 20s clock.

## 2. Hard Compliance Checks (Anti-Disqualification)
- [ ] **Haiku-only network check:** Zero calls to any hosted LLM other than the arena gateway (`llm["base_url"]`, model `haiku`).
- [ ] **Chat rule check:** Every `start` and `turn` sends `chat` (<=200 chars), opponent `chat` reaches the decision/model (marked untrusted), and Guardrails restrictions are respected.
- [ ] **No data exfiltration:** No game rules, observations, or chat sent to external servers; HTTP calls honor `HTTPS_PROXY` (`NO_PROXY` keeps `gateway,localhost,127.0.0.1` direct).
- [ ] **Concurrency-safe `/memory`:** Writes keyed by `game_id` + rules hash (atomic write/rename) since multiple containers share `/memory`.

## 3. Push & Freeze Verification (Rule 14.6 & 3.7)
1. Push image with robot account (single quotes around robot username containing `$`):
   `docker login <registry> -u '<robot user>' -p '<secret>'`
   `docker push <registry>/<team>/bot:<tag>`
2. Verify remote digest:
   `docker buildx imagetools inspect <registry>/<team>/bot:<tag>`
3. Commit and tag the exact matching source code in Git (with Dockerfile, code, and prompts) and verify judges have read access!
4. Optional: ensure `/logo.png` (<=1 MB, no SVG) exists in the image if desired.
