# DADS worktree setup (for future agents)

This repo is worked on via **multiple git worktrees** to keep the RTMA humidity MVP isolated from the
larger cube/spatial refactor.

## Current worktrees (expected)

Run this any time to confirm:

```bash
git worktree list
git branch -v
```

You should see something like:

- `/home/dgketchum/code/dads` → branch `cube`
  - Purpose: Phase 0 / cube + spatial refactor work (heavier churn).
- `/home/dgketchum/code/dads-mvp` → branch `mvp/rtma-humidity`
  - Purpose: RTMA humidity MVP (ship-focused, minimal refactors).

During agent development we may also use:

- `/tmp/dads-mvp-codex` → branch `mvp/rtma-humidity-codex`
  - Purpose: temporary sandbox worktree used by Codex to make changes (then cherry-picked into
    `/home/dgketchum/code/dads-mvp`).

## Workflow rules

- **Do not develop the MVP on `cube`** — it will make merges/review painful.
- Treat `mvp/rtma-humidity` as the canonical MVP branch; keep commits small and reviewable.
- When a change is genuinely shared between cube + MVP, prefer **`git cherry-pick`** of a focused commit
  rather than repeatedly merging worktrees/branches both directions.
- Don’t commit large data artifacts (GeoTIFF/Zarr/etc.). Commit scripts + manifests + provenance notes.

## Creating the MVP worktree (if missing)

```bash
git worktree add ../dads-mvp main
cd ../dads-mvp
git switch -c mvp/rtma-humidity
```

## Removing a worktree cleanly

```bash
git worktree remove /path/to/worktree
git worktree prune
```

