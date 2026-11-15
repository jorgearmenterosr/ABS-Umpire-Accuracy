# Findings: 2026 challenge behaviour (notebooks 20 and 21)

15 Nov 2026. Source: `challenges_enriched`, `challenges_remaining`,
`games` (2026 regular season, 10,557 challenges in 2,429 games). Every
rate is MLB's own adjudication (overturned or not); it measures player
judgment, not umpire error. 95% Wilson intervals throughout.

## Key findings

1. **Catchers challenge far better than batters.** Among players with
   15+ challenges: catchers 58.5% (86 players, 5,488 challenges),
   batters 49.3% (105 players, 2,393). No pitcher reached 15
   (109 pitchers, 179 challenges, 39.7%). The tails differ too: the
   best high-volume challengers are catchers (Realmuto 70/86, CI 72–88%;
   Stephenson 93/136; Goodman 77/113), the worst are batters (Naylor
   7/26, Castro 6/24, Rafaela 6/24).

2. **Challenge quality falls as the stakes of the moment rise.**
   By inning: 1st–3rd 60.8%, 4th–6th 55.4%, 7th–8th 51.5%, 9th 40.5%
   (extras 44.3%). By count before the pitch: 0-0 59.9%, 1-2/2-2/3-1
   ~47%, 3-2 41.2%. Intervals at the ends do not overlap.

3. **Use it or lose it — confirmed.** In the 9th inning and later,
   teams holding 2 challenges succeed 36.4% of the time (n = 731) vs
   45.0% with 1 left (n = 845). Early in the game inventory makes no
   difference (56.4% vs 55.2%). Late challenges are worse because
   players spend expiring challenges on long shots, not because late
   pitches are closer.

4. **Most challenges go unused.** 75.1% of team-games end with at
   least one challenge unused; 28.5% end with both. Only 24.9% of
   team-games exhaust the allowance.

5. **Team spread is real at the ends, noise in the middle.** CIN 61.6%
   (228/370) to PIT 47.0% (167/355); PIT's upper bound (52.2%) is
   below CIN's lower bound (56.6%). The middle twenty teams, MIN 54.7%
   to SD 50.1%, are indistinguishable from the league's 53.9% and
   from each other. Five teams lost more than they won (PIT, CLE, STL,
   BAL, CWS). Volume (1.8–2.8 per game; MIN 450, CWS 441, STL 285)
   does not predict success.

6. **No home advantage.** Home 53.3% (5,253), away 54.4% (5,304).

## What to think about

- Findings 1–3 are probably one mechanism seen three ways: batters
  challenge the pitch that just went against them, catchers challenge
  the pitch they saw. Test: role × count split (batters should drive
  the full-count collapse).
- Finding 4 reframes the allowance question: before asking whether
  three challenges would help, ask why two are not used before the 9th.
- None of this says anything about umpire accuracy yet. Jorge's
  grading (J0) will show how often the umpire was actually wrong on
  challenged and unchallenged pitches.
- One season, the first with the rule; behaviour may shift in 2027.
- Present team results as tiers, not a league table (±5 points per
  team). Rank players and teams by `ci_low`, not by rate.

## Outputs

`reports/tables/`: player_challengers_2026_all.csv,
player_challengers_2026_ranked.csv, team_challengers_2026.csv,
challenges_by_inning_2026.csv, challenges_by_count_2026.csv,
unused_challenges_2026.csv. Notebooks: 20_player_challengers.ipynb,
21_team_challengers.ipynb.

## Next (Owen)

Role × count cell in notebook 21; O1 accuracy by season and O2 umpire
rankings once `pitches_graded` exists; O5 umpire × challenge rates.
