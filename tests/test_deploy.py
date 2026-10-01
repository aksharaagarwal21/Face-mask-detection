import os
import subprocess
import sys

import pytest

import deploy_hf_space as deploy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_space_gets_the_app_but_not_tests_or_docs(tmp_path):
    files = deploy.stage_files(str(tmp_path))
    for needed in ("Dockerfile", "app.py", "models/mask_detector.keras",
                   "models/face_detection_yunet_2023mar.onnx", "templates/field.html", "README.md"):
        assert needed in files
    assert not [f for f in files if f.startswith(("tests/", "docs/", ".github/"))]
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "sdk: docker" in readme and "app_port: 5000" in readme


@pytest.mark.parametrize("repo_id,host", [
    ("aksharaagarwal21/face-mask-detection", "https://aksharaagarwal21-face-mask-detection.hf.space"),
    ("Some.User/My_Space", "https://some-user-my-space.hf.space"),
])
def test_space_host(repo_id, host):
    assert deploy.space_host(repo_id) == host


def test_space_env_gives_phones_the_public_link():
    # on a Space: SPACE_HOST and the FMD_ACCESS_KEY secret are set; app is imported by gunicorn
    code = ("import app; c = app.app.test_client(); "
            "r = c.get('/api/connect', environ_base={'REMOTE_ADDR': '10.1.2.3'}, headers={'X-Access-Key': 'k3y'}); "
            "print(r.status_code, r.get_json()['phone_urls'][0])")
    env = {**os.environ, "SPACE_HOST": "me-face-mask-detection.hf.space", "FMD_ACCESS_KEY": "k3y"}
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                         env=env, timeout=120)
    assert out.stdout.split()[-2:] == ["200", "https://me-face-mask-detection.hf.space/field?key=k3y"], \
        out.stdout + out.stderr
