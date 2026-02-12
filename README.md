# Margdrashti Models

Road damage detection models using YOLOv8.

## Development Workflow

### Making Changes

1. Create a feature branch:
   ```bash
   git checkout -b your-branch-name
   ```

2. Make your changes and commit:
   ```bash
   git add .
   git commit -m "Your commit message"
   ```

3. Push to GitHub:
   ```bash
   git push -u origin your-branch-name
   ```

4. Create Pull Request:
   ```bash
   ./scripts/create_pr.sh "PR Title" "PR Description"
   ```

   The script will either:
   - Create a PR automatically using GitHub CLI (if installed), OR
   - Provide a link to create the PR manually on GitHub

   **Either way, you'll always be able to create your PR!**

### Creating Pull Requests

#### Quick Method (RECOMMENDED)
```bash
./scripts/create_pr.sh "Your PR title"
```

The script automatically:
- Uses your last commit message as the PR title
- Generates a list of commits for the PR body
- Tries to create the PR via GitHub CLI
- If that fails, provides a clickable URL to create PR manually

#### Manual Method
1. Go to: https://github.com/Shubhamf0073/Margdrashti_models/pulls
2. Click "New Pull Request"
3. Select your branch
4. Fill in title and description
5. Click "Create Pull Request"

### Troubleshooting PR Creation

**Problem: Script doesn't create PR automatically**

Don't worry! The script will always provide a URL you can click to create the PR manually.
This URL opens GitHub with everything pre-filled - just review and click "Create Pull Request".

**Problem: Want to install GitHub CLI for automation**

```bash
# Install gh CLI (requires sudo)
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update && sudo apt install gh -y

# Authenticate
gh auth login
```

**Problem: PR already exists for this branch**

Solution: Just push new commits to the same branch:
```bash
git push
```

The existing PR will automatically update with your new commits!

## Training YOLOv8 Models

### Quick Start

```bash
python scripts/yolov8_detection/train_yolov8.py \
    --data data/pothole_crack_detection/data.yaml \
    --model yolov8n.pt \
    --augment
```

See `scripts/yolov8_detection/train_yolov8.py` for all available options.

## License

See LICENSE file for details.
