# pyre-unsafe
"""
deploy_cloud_run.py — Publish the app on Google Cloud Run (free tier) in your Google Cloud project

    gcloud auth login                         # once
    gcloud config set project <project-id>    # a project with billing enabled
    python deploy_cloud_run.py                # creates or updates the service face-mask-detection

What it does:
  - takes the committed code (git archive of HEAD, so uncommitted edits are
    never published) minus tests, docs and CI files, like deploy_hf_space.py,
  - enables the Cloud Run, Cloud Build and Artifact Registry APIs,
  - builds the image from the Dockerfile with Cloud Build and runs it with
    2 GiB of memory, no instance while idle and at most one instance, which
    keeps light use inside the free tier,
  - on the first deploy, generates an access key and sets it as FMD_ACCESS_KEY.
    The URL is public, but every page and API call needs the key, which is
    kept in .deploy/cloud_run.json (git-ignored),
  - sets FMD_PUBLIC_URL so the dashboard shows the phone link,
  - checks the live app: /api/model answers with the key and /field is locked without it.

Later runs update the code and keep the key (--new-key replaces it).
"""

import argparse
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import time

from deploy_hf_space import ROOT, extract_commit, http_status, run

STATE_PATH = os.path.join(ROOT, ".deploy", "cloud_run.json")
APIS = ("run.googleapis.com", "cloudbuild.googleapis.com", "artifactregistry.googleapis.com")

GCLOUD = None       # path to gcloud (gcloud.cmd on Windows), set in main()


def gcloud(*args, allow_fail=False):
    """gcloud's output. On failure: None if allow_fail, else print gcloud's error and exit."""
    out = subprocess.run([GCLOUD, *args], cwd=ROOT, capture_output=True,
                         text=True, encoding="utf-8", errors="replace")
    if out.returncode:
        if allow_fail:
            return None
        sys.exit(f"gcloud {' '.join(args[:3])} failed:\n{out.stderr.strip()}")
    return out.stdout.strip()


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


def main(args):
    global GCLOUD
    GCLOUD = shutil.which("gcloud")
    if not GCLOUD:
        sys.exit("gcloud not found. Install the Google Cloud SDK "
                 "(https://cloud.google.com/sdk/docs/install), then run `gcloud auth login`.")
    if not gcloud("auth", "list", "--filter=status:ACTIVE", "--format=value(account)"):
        sys.exit("Not logged in to Google Cloud. Run `gcloud auth login` first.")
    project = args.project or gcloud("config", "get-value", "project")
    if not project:
        sys.exit("No Google Cloud project. Run `gcloud config set project <project-id>` or pass --project.")
    where = ["--project", project, "--region", args.region]

    if run("git", "status", "--porcelain", "--untracked-files=no").strip():
        print("⚠ Uncommitted changes are not deployed: Cloud Run gets the last commit.")
    sha = run("git", "rev-parse", "--short", "HEAD").strip()

    print(f"Enabling Cloud Run, Cloud Build and Artifact Registry in project {project}...")
    gcloud("services", "enable", *APIS, "--project", project)

    url = gcloud("run", "services", "describe", args.name, *where,
                 "--format=value(status.url)", allow_fail=True)
    service_id = f"{project}/{args.region}/{args.name}"
    print(f"{'Updating' if url else 'Creating'} Cloud Run service {service_id}")

    state = load_state()
    if state.get("service") != service_id:
        state = {}
    key = args.access_key
    if key is None and (not url or args.new_key or not state.get("access_key")):
        key = secrets.token_urlsafe(12)

    deploy = ["run", "deploy", args.name, *where,
              "--allow-unauthenticated",                    # public URL; the access key guards every request
              "--memory", "2Gi", "--cpu", "1", "--cpu-boost",   # TensorFlow needs >1 GiB; boost speeds up cold starts
              "--min-instances", "0", "--max-instances", "1",   # free while idle; one instance holds the per-phone trackers
              "--timeout", "120", "--quiet"]
    if key:
        deploy += ["--update-env-vars", f"FMD_ACCESS_KEY={key}"]
        state["access_key"] = key
        print("Access key will be set as FMD_ACCESS_KEY")
    else:
        print("Access key unchanged")

    with tempfile.TemporaryDirectory() as tmp:
        files = extract_commit(tmp)
        print(f"Building and deploying {len(files)} files from commit {sha} "
              f"(the first build takes ~10 min)...")
        if subprocess.run([GCLOUD, *deploy, "--source", tmp], cwd=ROOT).returncode:
            sys.exit("Deploy failed: see the gcloud output above.")

    url = gcloud("run", "services", "describe", args.name, *where, "--format=value(status.url)")
    if state.get("url") != url:
        print(f"Setting FMD_PUBLIC_URL={url} so the dashboard shows the phone link...")
        gcloud("run", "services", "update", args.name, *where,
               "--update-env-vars", f"FMD_PUBLIC_URL={url}", "--quiet")
    state.update(service=service_id, url=url, commit=sha)
    save_state(state)

    key = state.get("access_key")
    print("Checking the live app (the first request starts the container, ~30 s)...")
    status = None
    for _ in range(12):
        status = http_status(f"{url}/api/model", key, timeout=90)
        if status == 200:
            break
        time.sleep(5)
    locked = http_status(f"{url}/field")
    print(f"  /api/model with key -> {status} (expect 200)")
    print(f"  /field without key -> {locked} (expect 401)")
    ok = status == 200 and locked == 401

    print("\n" + "=" * 70)
    print(f"  Console:     https://console.cloud.google.com/run/detail/{args.region}/{args.name}?project={project}")
    print(f"  App:         {url}")
    if key:
        print(f"  Phone link:  {url}/field?key={key}")
        print(f"  Access key:  {key}   (saved in .deploy/cloud_run.json; share only with your team)")
    print("=" * 70)
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy to Google Cloud Run")
    parser.add_argument("--name", default="face-mask-detection", help="Cloud Run service name")
    parser.add_argument("--project", default=None, help="Google Cloud project (default: gcloud's current project)")
    parser.add_argument("--region", default="asia-south1", help="Cloud Run region (default: Mumbai)")
    parser.add_argument("--access-key", default=None, help="Use this access key instead of generating one")
    parser.add_argument("--new-key", action="store_true", help="Generate a new access key (old links stop working)")
    sys.exit(0 if main(parser.parse_args()) else 1)
