# Publish to GitHub

From the repository directory:

```bash
git init -b main
git add .
git commit -m "Initial SRUL project release"
```

Then authenticate with GitHub CLI and create the remote repository:

```bash
gh auth login
gh repo create MohammadrezaTavasoli/srul-generative-modeling \
  --public \
  --source=. \
  --remote=origin \
  --push
```

Use `--private` instead of `--public` for a private repository.

If an empty remote repository already exists:

```bash
git remote add origin https://github.com/MohammadrezaTavasoli/srul-generative-modeling.git
git push -u origin main
```
