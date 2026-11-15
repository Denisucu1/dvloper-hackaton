"""
Analyze top teams' strategies from Round Robin on the live arena platform.
"""
import urllib.request
import json
import collections

def fetch_json(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode('utf-8'))

def main():
    print("=== 1. FETCHING CURRENT STANDINGS ===")
    standings = fetch_json('https://arena.hackaton-2026.dvloper.io/api/standings')
    teams = standings.get('teams', [])
    for i, t in enumerate(teams):
        tid = t.get('id', '')
        tname = t.get('name', tid)
        pts = t.get('points', 0)
        rr_pts = t.get('round_robin', 0)
        t1_pts = t.get('t1', 0)
        t2_pts = t.get('t2', 0)
        print(f"{i+1:2d}. {tid:16s} ({tname:16s}) total={pts:5.1f} | RR={rr_pts:4.1f} | T1={t1_pts:4.1f} | T2={t2_pts:4.1f}")

    print("\n=== 2. ANALYZING ROUND ROBIN (RR1) PAIRINGS ===")
    rr = fetch_json('https://arena.hackaton-2026.dvloper.io/api/roundrobins/rr1?soft=1')
    pairings = rr.get('roundrobin', {}).get('pairings', [])
    print(f"Total pairings: {len(pairings)}")

    # Track game performance per team
    # team -> game_id -> {wins, losses, ties}
    game_stats = collections.defaultdict(lambda: collections.defaultdict(lambda: {'wins': 0, 'losses': 0, 'ties': 0}))
    top_teams = ['alt', 'nyx', 'caramel-syrup', 'systematic', '0tknsleft', 'adn']

    for p in pairings:
        for b in p.get('boards', []):
            gid = b.get('game_id')
            for g in b.get('results', []):
                seats = g.get('seats', [])
                winner = g.get('winner')
                if len(seats) == 2:
                    s0, s1 = seats[0], seats[1]
                    if winner == s0:
                        game_stats[s0][gid]['wins'] += 1
                        game_stats[s1][gid]['losses'] += 1
                    elif winner == s1:
                        game_stats[s1][gid]['wins'] += 1
                        game_stats[s0][gid]['losses'] += 1
                    else:
                        game_stats[s0][gid]['ties'] += 1
                        game_stats[s1][gid]['ties'] += 1

    print("\n=== 3. GAME-BY-GAME WIN RATES FOR TOP TEAMS & ADN ===")
    all_games = sorted(list({gid for t in top_teams for gid in game_stats[t].keys()}))
    header = f"{'Team':15s}" + "".join([f"{gid[:8]:>9s}" for gid in all_games])
    print(header)
    print("-" * len(header))
    for t in top_teams:
        row = f"{t:15s}"
        for gid in all_games:
            w = game_stats[t][gid]['wins']
            l = game_stats[t][gid]['losses']
            tot = w + l
            if tot > 0:
                row += f"  {w}/{tot}   "
            else:
                row += "   -     "
        print(row)

    print("\n=== 4. INSPECTING DETAILED MATCH LOGS OF TOP TEAMS ===")
    # Pick a few key matches of A.L.T., NYX, Caramel Syrup to inspect actions and response times
    key_matches = [
        # ALT
        ('rr1-r1-m1', 'alt vs path-to-glory'),
        ('rr1-r5-m4', 'alt vs caramel-syrup'),
        ('rr1-r13-m1', 'alt vs adn'),
        # NYX
        ('rr1-r3-m2', 'nyx vs alt'),
        ('rr1-r12-m3', 'nyx vs caramel-syrup'),
        # Caramel Syrup
        ('rr1-r8-m4', 'caramel-syrup vs adn'),
    ]

    for mid, desc in key_matches:
        try:
            m_data = fetch_json(f'https://arena.hackaton-2026.dvloper.io/api/matches/{mid}')
            print(f"\n--- MATCH {mid} ({desc}) --- Score: {m_data.get('score')} Winner: {m_data.get('winner')}")
            for g in m_data.get('games', []):
                gid = g.get('game_id')
                vname = g.get('variant')
                seats = g.get('seats')
                res = g.get('result', {})
                timeline = g.get('timeline', [])
                
                # Check timing and moves for each seat
                seat_times = collections.defaultdict(list)
                seat_moves = collections.defaultdict(list)
                for step in timeline:
                    if step.get('type') == 'turn':
                        seat_idx = step.get('seat')
                        team_name = seats[seat_idx] if seat_idx is not None and seat_idx < len(seats) else f"seat_{seat_idx}"
                        dur = step.get('duration_ms', 0)
                        act = step.get('action')
                        chat = step.get('chat')
                        seat_times[team_name].append(dur)
                        seat_moves[team_name].append(act)

                print(f"  Game: {gid} ({vname}) | Winner: {res.get('winner')}")
                for tm, times in seat_times.items():
                    avg_t = sum(times) / len(times) if times else 0
                    max_t = max(times) if times else 0
                    print(f"    [{tm}] turns={len(times)}, avg_ms={avg_t:.1f}, max_ms={max_t:.1f}, sample_actions={seat_moves[tm][:4]}")
        except Exception as e:
            print(f"Error fetching {mid}: {e}")

if __name__ == '__main__':
    main()
