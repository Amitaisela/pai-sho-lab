# Contributing

Pai Sho Lab is an independent project: it has its own history and its own copy of the rules
engine and the agents. It is not generated from any other repository.

## Rules and the online site

The online site (where bots play each other, see "Play online" in the README) runs its own copy
of the engine. Both copies must agree on the rules, so every rules change is versioned:

- `Agents/elo.py` has `RULES_VERSION`. Bump it when a change alters which moves are legal or who wins.
- The site announces its `rules_version` at `GET /api/bot/meta` and in the first line of the event stream.
- The connector (`scripts/play_online.py`) refuses to play when the numbers differ and tells you to update your clone.

So if you change the rules: bump the version here and open an issue so the site's copy is updated to match.
Until the site matches, players on the new version cannot play online (they get a clear message), which is the intended failure.

## Adding an agent

See the README ("I want to add an agent"). One Python file plus one registry entry.

## Tests

`python tests/basic_tests.py`, `python tests/test_rules_contract.py`, `python tests/test_notation.py`,
`python tests/test_online_client.py`, and `cd engine/RustEngine && cargo test --workspace`.
`ruff check .` must pass.
