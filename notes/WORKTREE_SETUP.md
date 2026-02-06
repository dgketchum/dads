# DADS worktree setup (for future agents)

Canonical assumption for all work: **start from** `/home/dgketchum/code/dads`.

This repo uses **git worktrees** to keep the RTMA humidity MVP isolated from the larger cube/spatial refactor.

## Current worktrees (expected)

Run this any time to confirm:

```bash
cd /home/dgketchum/code/dads
git worktree list
git branch -v
```

You should see something like:

- `/home/dgketchum/code/dads` → branch `cube`
  - Purpose: Phase 0 / cube + spatial refactor work (heavier churn).
- `/home/dgketchum/code/dads-mvp` → branch `mvp/rtma-humidity`
  - Purpose: RTMA humidity MVP (ship-focused, minimal refactors).

## Workflow rules

- **Do not develop the MVP on `cube`** — it will make merges/review painful.
- Treat `mvp/rtma-humidity` as the canonical MVP branch; keep commits small and reviewable.
- When a change is genuinely shared between cube + MVP, prefer **`git cherry-pick`** of a focused commit
  rather than repeatedly merging worktrees/branches both directions.
- Don’t commit large data artifacts (GeoTIFF/Zarr/etc.). Commit scripts + manifests + provenance notes.

## Agent sandbox note (if you can’t write to `/home/dgketchum/code/dads-mvp`)

Some agent/sandbox environments can only write within a subset of directories.

If an agent can’t write to `/home/dgketchum/code/dads-mvp`, it may:
- create a temporary worktree under `/tmp` (or another writable location),
- make commits there,
- then `git cherry-pick` those commits into `/home/dgketchum/code/dads-mvp`.

This is an implementation detail for constrained environments; the *human* workflow stays: work from the two
local worktrees listed above.

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
