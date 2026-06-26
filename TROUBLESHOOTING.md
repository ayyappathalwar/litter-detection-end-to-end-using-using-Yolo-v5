# CI/CD Troubleshooting Guide
## Litter Detection - YOLOv5 End-to-End Deployment

This document captures every issue encountered during the CI/CD setup and deployment of this project, along with the exact fix applied. Use this as a reference if the same issues arise again.

---

## Issue 1: GitHub Actions Failing — Deprecated Action Versions

### Symptom
- Continuous Delivery job fails with `exit code 1`
- Error: `The process '/usr/bin/git' failed with exit code 128`
- Warning: `Node.js 20 is deprecated, being forced onto Node.js 24`

### Root Cause
The workflow used old versions of GitHub Actions that relied on Node.js 16/20. GitHub now forces these onto Node.js 24, causing compatibility failures.

Old versions used:
```yaml
uses: actions/checkout@v3                       # Node 16 — breaks on Node 24
uses: aws-actions/configure-aws-credentials@v1  # deprecated
uses: aws-actions/amazon-ecr-login@v1           # deprecated
```

### Fix
Update all action versions in `.github/workflows/main.yaml`:
```yaml
uses: actions/checkout@v4
uses: aws-actions/configure-aws-credentials@v4
uses: aws-actions/amazon-ecr-login@v2
```

---

## Issue 2: Deprecated `set-output` Syntax Causing Exit Code 1

### Symptom
- "Build, tag, and push image to Amazon ECR" step fails with `exit code 1`
- No clear Docker error, but the step always fails

### Root Cause
The workflow used the old GitHub Actions output syntax which was **disabled by GitHub in May 2024**:
```bash
echo "::set-output name=image::$ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG"
```

### Fix
Replace with the new syntax in `.github/workflows/main.yaml`:
```bash
echo "image=$ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG" >> $GITHUB_OUTPUT
```

---

## Issue 3: Wrong AWS Region

### Symptom
- Docker push to ECR fails
- ECR login succeeds but push is rejected

### Root Cause
The workflow had `aws-region: eu-north-1` but the EC2 instance and ECR repository were in `ap-south-1` (Mumbai).

### Fix
Update all region references in `.github/workflows/main.yaml`:
```yaml
aws-region: ap-south-1
```
Also update the `AWS_REGION` environment variable passed to the Docker container:
```bash
-e AWS_REGION=ap-south-1
```

> **Important:** Also verify the `AWS_ECR_LOGIN_URI` GitHub Secret contains the correct region in the URI format:
> `<account-id>.dkr.ecr.ap-south-1.amazonaws.com`

---

## Issue 4: Self-Hosted Runner Conflict

### Symptom
EC2 runner shows:
```
A session for this runner already exists.
Runner connect error: Error: Conflict. Retrying until reconnected.
```

### Root Cause
Two instances of `./run.sh` are running simultaneously — one from a previous session and one newly started.

### Fix
Kill the existing runner process and restart cleanly:
```bash
pkill -f Runner.Listener
# Wait 3 seconds
./run.sh
```

### Prevention
Set up the runner as a systemd service so it starts automatically and only runs one instance:
```bash
sudo ./svc.sh install
sudo ./svc.sh start
```

---

## Issue 5: Container Crashes — `ModuleNotFoundError: No module named 'wasteDetection'`

### Symptom
```
docker logs waste
ModuleNotFoundError: No module named 'wasteDetection'
Exited (1)
```

### Root Cause
The `-e .` editable install in `requirements.txt` does not work reliably inside Docker builds. The `wasteDetection` package was never registered in Python's site-packages.

### Fix
**Step 1:** Remove `-e .` from `requirements.txt`

**Step 2:** Add explicit package installation in `Dockerfile`:
```dockerfile
RUN pip install -r requirements.txt && pip install .
```

**Step 3:** Add `PYTHONPATH` as a safety net:
```dockerfile
ENV PYTHONPATH=/app
```

---

## Issue 6: Case Sensitivity — `WasteDetection` vs `wasteDetection`

### Symptom
Even with `PYTHONPATH=/app` set, still getting:
```
ModuleNotFoundError: No module named 'wasteDetection'
```

Running `docker run ... ls /app` shows the directory as `WasteDetection` (capital W), but `app.py` imports `wasteDetection` (lowercase w).

### Root Cause
The project was developed on **Windows** (case-insensitive filesystem). Git stored the directory as `WasteDetection`. When built on Linux (case-sensitive), the directory is `WasteDetection` but imports use `wasteDetection` — Python can't find it.

### Fix
Add a symlink in the `Dockerfile` so both casings point to the same directory:
```dockerfile
RUN ln -sf /app/WasteDetection /app/wasteDetection
```

---

## Issue 7: `best.pt` Model Weights Missing from Docker Image

### Symptom
```
docker logs waste
YOLOv5 detect.py exited with code 2
# or
best.pt: No such file or directory
```

### Root Cause
`best.pt` is inside the `yolov5/` folder which is a git submodule. GitHub Actions does not check out submodule contents by default. So the model weights were never included in the Docker image.

### Fix
Copy `best.pt` to the **root of the repository** (outside the submodule):
```bash
cp yolov5/best.pt best.pt
git add best.pt
git commit -m "add best.pt model weights"
git push origin main
```

Then copy it into the correct location in the `Dockerfile`:
```dockerfile
COPY . /app
# ... other steps ...
RUN cp /app/best.pt /app/yolov5/best.pt
```

---

## Issue 8: `detect.py` Missing — YOLOv5 Submodule Not Checked Out

### Symptom
```
docker logs waste
python: can't open file 'detect.py': [Errno 2] No such file or directory
CMD FAILED (exit 2)
```

### Root Cause
The `yolov5/` directory is a **broken git submodule** — git tracks it as a submodule but there is no `.gitmodules` file. GitHub Actions cannot check out its contents. As a result, the entire `yolov5/` folder (including `detect.py`) is empty in the Docker image.

### What Doesn't Work
Adding `submodules: true` to the checkout step fails because `.gitmodules` is missing:
```
fatal: no submodule mapping found in .gitmodules for path 'yolov5'
```

### Fix
Clone YOLOv5 fresh from GitHub inside the `Dockerfile`:
```dockerfile
RUN rm -rf /app/yolov5 && git clone https://github.com/ultralytics/yolov5.git /app/yolov5
```

> **Note:** `rm -rf /app/yolov5` is required first because `COPY . /app` creates an empty `yolov5/` directory (the broken submodule placeholder). `git clone` refuses to clone into an existing directory.

---

## Issue 9: `NotImplementedError: cannot instantiate 'WindowsPath' on your system`

### Symptom
```
docker logs waste
File "/usr/local/lib/python3.8/pathlib.py", line 1044, in __new__
    raise NotImplementedError("cannot instantiate %r on your system"
NotImplementedError: cannot instantiate 'WindowsPath' on your system
CMD FAILED (exit 1)
```

### Root Cause
`best.pt` was **trained and saved on Windows**. PyTorch's `torch.save` pickles Python objects including `pathlib.WindowsPath` references. When the model is loaded on Linux, `WindowsPath` cannot be instantiated because it is a Windows-only class.

### Fix
Create a `sitecustomize.py` file in the Docker image. Python automatically executes this file before anything else runs — it patches `WindowsPath` globally for all scripts including `detect.py`:

```dockerfile
RUN echo "import pathlib; pathlib.WindowsPath = pathlib.PosixPath" > /usr/local/lib/python3.8/sitecustomize.py
```

> **Why not `torch.load` during build?** Attempting to re-save the model during Docker build (`torch.load` + `torch.save`) caused the Docker build itself to fail, adding 12+ minutes to build time. The `sitecustomize.py` approach is instant and works for all future model loads.

---

## Issue 10: GitHub Actions Pipeline Not Triggering

### Symptom
A commit is visible on GitHub but no new workflow run appears in the Actions tab.

### Root Cause
Multiple rapid pushes in quick succession can cause GitHub Actions to skip triggering a run for intermediate commits.

### Fix
Push an empty commit to force a new pipeline run:
```bash
git commit --allow-empty -m "ci: trigger pipeline"
git push origin main
```

---

## Final Working Dockerfile

```dockerfile
FROM python:3.8-slim-bullseye
WORKDIR /app
COPY . /app

RUN apt update -y && apt install awscli git -y

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 unzip -y \
    && rm -rf /app/yolov5 && git clone https://github.com/ultralytics/yolov5.git /app/yolov5 \
    && pip install -r requirements.txt \
    && pip install . \
    && ln -sf /app/WasteDetection /app/wasteDetection \
    && cp /app/best.pt /app/yolov5/best.pt \
    && echo "import pathlib; pathlib.WindowsPath = pathlib.PosixPath" > /usr/local/lib/python3.8/sitecustomize.py

ENV PYTHONPATH=/app
EXPOSE 8081
CMD ["python3", "app.py"]
```

---

## Final Working GitHub Actions Workflow

```yaml
name: workflow

on:
  push:
    branches:
      - main
    paths-ignore:
      - 'README.md'

permissions:
  id-token: write
  contents: read

jobs:
  integration:
    name: Continuous Integration
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Code
        uses: actions/checkout@v4
      - name: Lint code
        run: echo "Linting repository"
      - name: Run unit tests
        run: echo "Running unit tests"

  build-and-push-ecr-image:
    name: Continuous Delivery
    needs: integration
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Code
        uses: actions/checkout@v4
      - name: Install Utilities
        run: |
          sudo apt-get update
          sudo apt-get install -y jq unzip
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ap-south-1
      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v2
      - name: Build, tag, and push image to Amazon ECR
        id: build-image
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          ECR_REPOSITORY: ${{ secrets.ECR_REPOSITORY_NAME }}
          IMAGE_TAG: latest
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
          echo "image=$ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG" >> $GITHUB_OUTPUT

  Continuous-Deployment:
    needs: build-and-push-ecr-image
    runs-on: self-hosted
    steps:
      - name: Checkout
        uses: actions/checkout@v4
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ap-south-1
      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v2
      - name: Stop and remove container if running
        run: |
          docker ps -q --filter "name=waste" | grep -q . && docker stop waste && docker rm -fv waste || true
      - name: Clean previous images and containers
        run: |
          docker system prune -af
      - name: Pull latest images
        run: |
          docker pull ${{ secrets.AWS_ECR_LOGIN_URI }}/${{ secrets.ECR_REPOSITORY_NAME }}:latest
      - name: Run Docker Image to serve users
        run: |
          docker run -d -p 8081:8081 --ipc="host" --name=waste \
            -e AWS_ACCESS_KEY_ID=${{ secrets.AWS_ACCESS_KEY_ID }} \
            -e AWS_SECRET_ACCESS_KEY=${{ secrets.AWS_SECRET_ACCESS_KEY }} \
            -e AWS_REGION=ap-south-1 \
            ${{ secrets.AWS_ECR_LOGIN_URI }}/${{ secrets.ECR_REPOSITORY_NAME }}:latest
```

---

## Quick Diagnostic Commands (EC2)

```bash
# Check if container is running
docker ps -a --filter "name=waste"

# View container logs
docker logs waste --tail 50

# Check if best.pt exists inside container
docker exec waste ls -lh /app/yolov5/best.pt

# Check what files are in /app inside container
docker run --rm <image-uri> ls /app

# Test if port 8081 is open (run locally)
curl -I http://<EC2-PUBLIC-IP>:8081/

# Restart runner (if conflict error)
pkill -f Runner.Listener
cd ~/actions-runner && ./run.sh
```

---

## Security Reminder

- **Never expose AWS credentials in chat or logs.** If accidentally exposed, immediately go to **AWS Console → IAM → Users → Security credentials → deactivate & delete** the key and create a new one.
- Update the new credentials in **GitHub → Settings → Secrets**.
