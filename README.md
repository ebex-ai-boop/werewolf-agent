# Werewolf agent — make.fun werewolf-agent-01

A model-free, rule-based Werewolf agent. Python 3.13, standard library
only, deterministic: every decision is a pure function of the env view
(no clock, no unseeded randomness), so a trial replays exactly.

## How it plays

The agent keeps a suspicion score for every living seat, rebuilt from the
full public log (votes, speeches, deaths, revealed roles) plus its own
private knowledge (seer checks, wolf partner, last protection):

- **Villager / doctor** — votes the most suspicious seat, joining the
  plurality instead of splitting the vote; abstains when there is no real
  signal rather than exiling a villager on a guess. The doctor protects
  the credible seer claimant first.
- **Seer** — inspects the most suspicious uninspected seat and reveals a
  found wolf in debate at once, so the village can follow the evidence.
  Claims are checked against revealed roles: a seer claim proven false
  exposes a lying wolf, one proven true certifies the real seer.
- **Werewolf** — the kill only lands when both wolves name the same seat,
  so the second wolf always matches the partner's night vote; the first
  wolf targets the claimed seer or the most accurate accuser. By day it
  votes with the village consensus and accuses a villager, never its
  partner (it only buses the partner when the exile is already decided).

## Running

The program is run by the grader as `python3 -P -s /bot/main.py`. It reads
`MAKEFUN_ENV_URL` (the game env) and `MAKEFUN_TRIAL_SEED` (any randomness
it needs, seeded per decision), polls `POST /observe`, answers each
decision with `POST /act`, and exits once the trial reports `done`.

## Local testing

`work/sim.py` in the build workspace replays the kit's practice deals
against scripted house players and reports wolf/village win rates and the
score. `work/ollama_sim.py` replays two games with real model house
players (Ollama qwen3:14b on the builder's PC) for the debate and vote
decisions of days 1-3.

## License

MIT — see `LICENSE`.
