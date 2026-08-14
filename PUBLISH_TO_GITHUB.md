# Publish this repository to GitHub

The repository is already initialized on the `main` branch with an initial commit.

## GitHub CLI

Install and authenticate GitHub CLI, then run from this directory:

```bash
gh auth login
gh repo create MohammadrezaTavasoli/srul-generative-modeling --public --source=. --remote=origin --push
```

Use `--private` instead of `--public` if private visibility is required.

## Existing empty GitHub repository

After creating an empty repository named `srul-generative-modeling` on GitHub:

```bash
git remote add origin https://github.com/MohammadrezaTavasoli/srul-generative-modeling.git
git push -u origin main
```
