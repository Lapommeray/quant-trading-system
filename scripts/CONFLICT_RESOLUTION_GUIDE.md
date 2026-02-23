# Resolve conflicts between `codex/locate-each-folder-individually` and `main`

This is a step-by-step checklist to resolve branch conflicts and create/update a PR.

## 1) In GitHub UI: verify branch and open conflict view

1. Go to: `https://github.com/Lapommeray/quant-trading-system`
2. Click **Pull requests**.
3. Click the PR for `codex/locate-each-folder-individually` (or click **New pull request** and set base=`main`, compare=`codex/locate-each-folder-individually`).
4. If GitHub shows **This branch has conflicts that must be resolved**, click **Resolve conflicts**.

## 2) In GitHub UI: resolve directly (small conflicts only)

1. In the conflict editor, for each file:
   - Keep the intended final logic (remove `<<<<<<<`, `=======`, `>>>>>>>` markers).
   - Ensure imports still compile and tests remain deterministic.
2. Click **Mark as resolved** for each file.
3. Click **Commit merge**.

If conflicts are too large for UI, use local CLI workflow below.

## 3) Local CLI conflict resolution (recommended for multi-file conflicts)

From repository root:

```bash
git checkout codex/locate-each-folder-individually
git fetch origin
git merge origin/main
```

Then resolve each conflicted file shown by:

```bash
git status
```

After editing files:

```bash
git add <file1> <file2> ...
git commit -m "Resolve merge conflicts with main for codex/locate-each-folder-individually"
git push -u origin codex/locate-each-folder-individually
```

## 4) Validate before merge

```bash
PYTHONPATH=. pytest -q
```

Expected: tests pass/skip without collection crashes.

## 5) Finalize PR

1. Return to **Pull requests**.
2. Verify the PR is now mergeable (no conflicts).
3. Confirm checks are green.
4. Click **Merge pull request**.

## Notes for this environment

If direct push from this runtime is blocked by proxy/network policy, use the repository's manual GitHub Action workflow to push/open PR from CI auth:

- Workflow: `.github/workflows/push_and_open_pr.yml`
- Guide: `scripts/push_and_open_pr_ci.md`
