# Contributing

This repo is reviewed by people and by AI agents. The rules below exist so a
reviewer can audit any commit on its own, re-run its verification, and trust the
result without a GPU.

## Commits

- **One concern per commit, green at every commit.** `make check` (ruff + pytest,
  CPU only) passes on every commit on every branch. No WIP commits that get
  squashed later; if a change needs two steps to be green, it is two commits.
- **Data and code never share a commit.** Request fixtures, generated oracle
  outputs, and scripts land in separate commits so logic can be diffed without
  generated JSON in the way.
- **Message format:**

  ```
  <area>: <imperative summary, ≤ 60 chars>

  Why: <the problem or need, one or two sentences>
  What: <what changed, in terms a reviewer can check>
  Verified: <exact command> → <one-line result>
  ```

  `<area>` is a top-level path (`scripts`, `deciders/clef`, `fixtures`, `ci`,
  `docs`). Trailers for co-authorship go after a blank line.
- **Tests run without a GPU.** Anything that can be unit-tested is tested against
  a tiny synthetic fixture under `tests/`. GPU-only code ships a `--dry-run` that
  exercises everything except the forward pass.

## Branches and pull requests

- `main` only moves by pull request.
- Generated results (oracle outputs, benchmark runs) are produced by a pod and
  pushed to `oracle/<model>-<rev7>-<yyyymmdd>` or `bench/<arm>-<yyyymmdd>`. The
  PR description carries the run's `manifest.json` and the parity report.
- Each milestone has a `REVIEW.md` section: what to run, what output to expect,
  what is deliberately out of scope. That section is what an auditing agent is
  handed.

## Verification before building

Interfaces are checked with a cheap command before code depends on them (an
import, `pip index versions`, `nvidia-smi`, a one-record run). The check and its
output go in the commit's `Verified:` line.
