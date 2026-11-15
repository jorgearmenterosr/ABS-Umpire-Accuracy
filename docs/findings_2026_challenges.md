# Findings: 2026 challenge behaviour (notebooks 20, 21, 22)

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

7. **Players challenge worst on the calls that hurt their own side**
   (notebook 22). Batters: 64% at 1-0/2-0, 40% with two strikes.
   Catchers: 63-69% at 0-0/0-1, 45-48% at 3-0/3-1/3-2. Each role's
   worst counts are the ones where the call goes against it. 3-2 is
   the busiest and worst count for both (495 / 471 challenges; 38% /
   45%).

8. **Catchers burn the spare challenge too.** 9th inning+ with 2 left:
   catchers 36% (n 389), batters 37% (n 336). With 1 left: catchers
   51%, batters 41%. Early game: catchers 61% regardless. The catcher
   advantage holds only when a challenge is scarce.

9. **Pitch type.** Curveballs 60% (786), changeups 57%, sliders /
   cutters / sinkers 54-55%, four-seamers 53% (3,569, a third of all),
   sweepers 49% (815). By volume: FF 34%, SI 21%, SL 12%.

10. **The 2026 called zone sits above the ABS zone** (first
    umpire-behaviour finding, from challenge outcomes alone). Called
    balls in the low third overturned 79%; called strikes there 22%.
    Called strikes above the zone overturned 64%; called balls above
    49%. Umpires miss low strikes and give high ones. Challenged
    pitches only; Jorge's J3 contour will put inches on it.

## What to think about

- Findings 1-3 and 7-8 are one mechanism: incentives. Players
  challenge worst when the call hurts them most, and everyone burns a
  spare challenge in the 9th. The catcher advantage is conditional on
  scarcity, not innate.
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
unused_challenges_2026.csv, challenges_role_by_count_2026.csv,
challenges_role_by_late_inventory_2026.csv,
challenges_by_pitch_type_2026.csv,
challenges_by_call_and_location_2026.csv. Notebooks: 20, 21, 22.

## Next (Owen)

O1 accuracy by season and O2 umpire
rankings once `pitches_graded` exists; O5 umpire × challenge rates.
