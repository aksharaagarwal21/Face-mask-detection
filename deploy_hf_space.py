# pyre-unsafe
"""
deploy_hf_space.py — Publish the app as a Hugging Face Space (Docker) under your account

    hf auth login                    # once, with a Write token
    python deploy_hf_space.py        # creates or updates spaces/<you>/face-mask-detection

What it does:
  - takes the committed code (git archive of HEAD, so uncommitted edits are
    never published) minus tests, docs and CI files,
  - adds the Space README (Docker SDK, port 5000),
  - creates the Space if needed and uploads the files; Hugging Face builds the
    image from the Dockerfile on its free CPU hardware,
  - on the first deploy, generates an access key and stores it as the Space
    secret FMD_ACCESS_KEY. The Space is public, but every page and API call
    needs the key, which is kept in .deploy/hf_space.json (git-ignored),
  - waits for the build and checks the live app: /api/model answers with the
    key and /field is locked without it.

Later runs update the code and keep the key (--new-key replaces it).
"""

import argparse
import io
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request

for _stream in (sys.stdout, sys.stderr):        # Windows consoles: print ⚠ and emoji safely
    if hasattr(_stream, "reconfigure") and (_stream.encoding or "").lower() not in ("utf-8", "utf8"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.abspath(__file__))
STATE_PATH = os.path.join(ROOT, ".deploy", "hf_space.json")
GITHUB_URL = "https://github.com/aksharaagarwal21/Face-mask-detection"

# Not needed to run the app
EXCLUDE = ("tests/", "docs/", ".github/", "models/plots/", "README.md", "pytest.ini",
           "requirements-dev.txt", ".gitignore")

SPACE_README = """---
title: Face Mask Detection
emoji: 😷
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 5000
pinned: false
short_description: Check mask compliance from a phone camera, up close or far
---

# Face Mask Detection

Detects faces and classifies each as **with mask**, **without mask** or
**mask worn incorrectly** (98.97% accuracy on held-out test faces).

- **Field mode** (`/field`) for phones: back camera, zoom for distant people,
  enlarged thumbnails of flagged faces, freeze/photo/save
- **Dashboard** (`/`) with photo analysis and model metrics
- **REST API**: `POST /api/predict` with an image

**Access:** this Space requires an access key. Open the link you were given
(`/field?key=...`) or send the key as an `X-Access-Key` header. Photos are
analysed in memory and not stored.

Source, model card and results: {github}
"""


def run(*cmd):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True).stdout


def extract_commit(dest):
    """Committed files (HEAD) needed to run the app, extracted into dest. Returns the file list."""
    archive = subprocess.run(["git", "archive", "--format=tar", "HEAD"], cwd=ROOT,
                             capture_output=True, check=True).stdout
    kept = []
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar.getmembers():
            if not member.isfile() or member.name.startswith(EXCLUDE):
                continue
            tar.extract(member, dest)
            kept.append(member.name)
    return kept


def stage_files(dest):
    """The app files plus the Space README, extracted into dest. Returns the file list."""
    kept = extract_commit(dest)
    with open(os.path.join(dest, "README.md"), "w", encoding="utf-8") as f:
        f.write(SPACE_README.format(github=GITHUB_URL))
    return sorted(kept + ["README.md"])


def space_host(repo_id):
    """https://<owner>-<name>.hf.space"""
    return "https://" + re.sub(r"[^a-z0-9]+", "-", repo_id.replace("/", "-").lower()).strip("-") + ".hf.space"


def load_state():
    try:
        with open(STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_state(state):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def http_status(url, key=None, timeout=30):
    req = urllib.request.Request(url, headers={"X-Access-Key": key} if key else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except OSError:
        return None


def wait_for_space(api, repo_id, timeout_s):
    """Poll the Space until it runs (True) or the build/start fails (False)."""
    t0, last = time.time(), None
    while time.time() - t0 < timeout_s:
        stage = api.get_space_runtime(repo_id).stage
        if stage != last:
            print(f"  [{int(time.time() - t0):>4}s] {stage}")
            last = stage
        if stage == "RUNNING":
            return True
        if stage in ("BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "NO_APP_FILE"):
            return False
        time.sleep(15)
    print("  timed out waiting; check the Space page for build logs")
    return False


def main(args):
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        user = api.whoami()["name"]
    except Exception:
        sys.exit("Not logged in to Hugging Face. Run `hf auth login` with a Write token first.")
    repo_id = f"{args.owner or user}/{args.name}"
    host = space_host(repo_id)

    if run("git", "status", "--porcelain", "--untracked-files=no").strip():
        print("⚠ Uncommitted changes are not deployed: the Space gets the last commit.")
    sha = run("git", "rev-parse", "--short", "HEAD").strip()

    ci = bool(os.environ.get("CI"))     # GitHub Actions: logs are public, never print the key
    is_new = not api.repo_exists(repo_id, repo_type="space")
    if ci and is_new:
        sys.exit(f"{repo_id} doesn't exist yet. Run `python deploy_hf_space.py` on your own computer "
                 "once, so the access key is created where you can see it.")
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)
    print(f"{'Created' if is_new else 'Updating'} Space {repo_id}")

    state = load_state() if load_state().get("space") == repo_id else {}
    key = args.access_key
    if key is None and (is_new or args.new_key or (not state.get("access_key") and not ci)):
        key = secrets.token_urlsafe(12)
    if key:
        api.add_space_secret(repo_id, "FMD_ACCESS_KEY", key,
                             description="Access key every visitor must present (see app.py)")
        state["access_key"] = key
        print("Access key set as Space secret FMD_ACCESS_KEY")
    elif not ci:
        print("Access key unchanged" + ("" if state.get("access_key") else
              " (not known on this machine: use --new-key to set a new one)"))

    with tempfile.TemporaryDirectory() as tmp:
        files = stage_files(tmp)
        print(f"Uploading {len(files)} files from commit {sha}...")
        api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=tmp,
                          commit_message=f"Deploy {sha} from {GITHUB_URL}",
                          delete_patterns=["*.py", "static/*", "templates/*", "models/*"])

    state.update(space=repo_id, url=host, commit=sha)
    if not ci:
        save_state(state)

    ok = True
    if not args.no_wait:
        print("Waiting for Hugging Face to build and start the Space (first build ~10 min)...")
        ok = wait_for_space(api, repo_id, args.timeout)
        if ok:
            key = state.get("access_key")
            locked = http_status(f"{host}/field")
            print(f"  /field without key -> {locked} (expect 401)")
            if key:
                for _ in range(10):
                    status = http_status(f"{host}/api/model", key)
                    if status == 200:
                        break
                    time.sleep(6)
                print(f"  /api/model with key -> {status} (expect 200)")
                ok = status == 200 and locked == 401

    key = state.get("access_key")
    print("\n" + "=" * 70)
    print(f"  Space page:  https://huggingface.co/spaces/{repo_id}")
    print(f"  App:         {host}")
    if key and not ci:
        print(f"  Phone link:  {host}/field?key={key}")
        print(f"  Access key:  {key}   (saved in .deploy/hf_space.json; share only with your team)")
    print("=" * 70)
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy to a Hugging Face Space")
    parser.add_argument("--name", default="face-mask-detection", help="Space name")
    parser.add_argument("--owner", default=None, help="User or organisation (default: you)")
    parser.add_argument("--private", action="store_true",
                        help="Private Space: only signed-in members can open it")
    parser.add_argument("--access-key", default=None, help="Use this access key instead of generating one")
    parser.add_argument("--new-key", action="store_true", help="Generate a new access key (old links stop working)")
    parser.add_argument("--no-wait", action="store_true", help="Don't wait for the build")
    parser.add_argument("--timeout", type=int, default=1800, help="Seconds to wait for the build")
    sys.exit(0 if main(parser.parse_args()) else 1)
