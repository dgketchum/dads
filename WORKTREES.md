# Worktree Layout

Use two git worktrees so the RTMA humidity MVP can iterate quickly without getting tangled in the cube/spatial refactor.

## Recommended directories

- `/home/dgketchum/code/dads` → branch `cube`
  - Phase 0+ cube/spatial work (higher churn).
- `/home/dgketchum/code/dads-mvp` → branch `mvp/rtma-humidity`
  - RTMA humidity MVP (ship-focused).

Confirm at any time (run from the canonical worktree):

```bash
git -C /home/dgketchum/code/dads worktree list
git -C /home/dgketchum/code/dads branch -v
```

## Creating the MVP worktree

```bash
cd /home/dgketchum/code/dads
git worktree add ../dads-mvp main
cd ../dads-mvp
git switch -c mvp/rtma-humidity
```

## Cleaning up

```bash
cd /home/dgketchum/code/dads
git worktree remove ../dads-mvp
git worktree prune
```
