# pyre-unsafe
"""
tunnel.py — A public HTTPS link to this computer through Cloudflare's free quick tunnel

    winget install Cloudflare.cloudflared      # once
    python app.py --tunnel

cloudflared connects out to Cloudflare, which gives a random
https://<words>.trycloudflare.com address and forwards its requests to the app
on this computer. No account, card or open router port is needed. The address
changes every run and only works while the app runs. Phones allow the camera
on it because it's HTTPS with a real certificate.
"""

import atexit
import os
import re
import shutil
import subprocess
import time

URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
# Where the winget/MSI installer puts it, for terminals opened before it was added to PATH
KNOWN_PATHS = (r"C:\Program Files (x86)\cloudflared\cloudflared.exe",
               r"C:\Program Files\cloudflared\cloudflared.exe")


def find_cloudflared():
    return shutil.which("cloudflared") or next((p for p in KNOWN_PATHS if os.path.exists(p)), None)


def tunnel_url(log_text):
    """The trycloudflare.com address in cloudflared's log, or None."""
    m = URL_RE.search(log_text)
    return m.group(0) if m else None


def start_quick_tunnel(port, log_path, timeout_s=60):
    """
    Start cloudflared forwarding to http://127.0.0.1:<port> and return the public
    URL. cloudflared stops when Python exits. RuntimeError if it can't start.
    """
    exe = find_cloudflared()
    if not exe:
        raise RuntimeError("cloudflared not found. Install it with: winget install Cloudflare.cloudflared")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen([exe, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
                            stdout=log, stderr=subprocess.STDOUT)
    atexit.register(proc.terminate)

    t0 = time.time()
    while time.time() - t0 < timeout_s and proc.poll() is None:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            url = tunnel_url(f.read())
        if url:
            return url
        time.sleep(0.5)
    proc.terminate()
    raise RuntimeError(f"cloudflared didn't report a tunnel address; see {log_path}")
