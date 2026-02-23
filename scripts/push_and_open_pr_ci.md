# Push and open PR in CI (GitHub-authenticated)

Use the `Push Branch And Open PR` workflow (`.github/workflows/push_and_open_pr.yml`) when local environment push is blocked.

## Steps
1. Go to **Actions** → **Push Branch And Open PR**.
2. Click **Run workflow**.
3. Provide:
   - `head_branch` (for example: `work`)
   - `base_branch` (for example: `main`)
   - `pr_title`
   - `pr_body`
4. Run.

The workflow will:
- push the selected branch to `origin`
- open/update the pull request using GitHub Actions `GITHUB_TOKEN`
