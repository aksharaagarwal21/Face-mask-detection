# pyre-unsafe
"""
secure_access.py — Let phones on the same network use the app: HTTPS + access key

Phones only allow live camera access (getUserMedia) on HTTPS pages, and a
server that can stream a webcam must not be open to everyone on the Wi-Fi.
This module:

  - finds this computer's address on the local network,
  - creates a self-signed certificate covering it (certs/, git-ignored; the
    phone shows a one-time warning that has to be accepted),
  - requires an access key on every request from another device. The key
    comes as ?key=..., an X-Access-Key header, or the cookie that is set
    after the first request with a valid ?key=. Requests from this
    computer itself (loopback) don't need it.
"""

import datetime
import hmac
import ipaddress
import os
import secrets
import socket

from flask import g, jsonify, request, Response

COOKIE_NAME = "fmd_key"
COOKIE_MAX_AGE = 12 * 3600          # one shift


def lan_ip():
    """This computer's address on the local network (127.0.0.1 if offline)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))      # UDP connect sends nothing; it only picks a route
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def new_access_key():
    return secrets.token_urlsafe(9)           # 12 URL-safe characters, 72 bits


# ─── Self-signed certificate ──────────────────────────────────────────────────
def _cert_ips(cert_path):
    from cryptography import x509
    with open(cert_path, "rb") as f:
        cert = x509.load_pem_x509_certificate(f.read())
    try:
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    except x509.ExtensionNotFound:
        return set(), cert
    return {str(ip) for ip in san.get_values_for_type(x509.IPAddress)}, cert


def ensure_certificate(cert_dir, ips):
    """
    (cert_path, key_path) for a self-signed certificate valid for localhost
    and `ips`. An existing certificate is reused while it covers all of them
    and has more than a week left.
    """
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    os.makedirs(cert_dir, exist_ok=True)
    cert_path = os.path.join(cert_dir, "server.crt")
    key_path = os.path.join(cert_dir, "server.key")
    wanted = {"127.0.0.1", *ips}
    now = datetime.datetime.now(datetime.timezone.utc)

    if os.path.exists(cert_path) and os.path.exists(key_path):
        have, cert = _cert_ips(cert_path)
        if wanted <= have and cert.not_valid_after_utc - now > datetime.timedelta(days=7):
            return cert_path, key_path

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Face Mask Detection (local)")])
    san = [x509.DNSName("localhost")] + [x509.IPAddress(ipaddress.ip_address(ip)) for ip in sorted(wanted)]
    cert = (x509.CertificateBuilder()
            .subject_name(name).issuer_name(name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=365))
            .add_extension(x509.SubjectAlternativeName(san), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .sign(key, hashes.SHA256()))
    with open(key_path, "wb") as f:
        f.write(key.private_bytes(serialization.Encoding.PEM,
                                  serialization.PrivateFormat.PKCS8,
                                  serialization.NoEncryption()))
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    return cert_path, key_path


# ─── Access key ───────────────────────────────────────────────────────────────
LOCKED_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Access key required</title>
<style>
  body { margin: 0; min-height: 100vh; display: grid; place-items: center;
         background: #0a0d14; color: #f1f5f9; font: 16px system-ui, sans-serif; }
  form { width: min(90vw, 360px); padding: 24px; border-radius: 14px;
         background: #111827; border: 1px solid rgba(255,255,255,.1); }
  h1 { font-size: 20px; margin: 0 0 8px; }
  p { color: #94a3b8; font-size: 14px; line-height: 1.4; }
  input, button { width: 100%; box-sizing: border-box; padding: 14px; font-size: 16px;
                  border-radius: 10px; border: 1px solid rgba(255,255,255,.15); }
  input { background: #0a0d14; color: #f1f5f9; margin: 8px 0 12px; }
  button { background: #0ea5e9; color: #001018; font-weight: 700; border: 0; }
</style></head>
<body><form method="get">
  <h1>Access key required</h1>
  <p>Scan the QR code on the computer running the detector, or type the key
     printed in its console.</p>
  <input name="key" autocomplete="off" autocapitalize="off" spellcheck="false"
         placeholder="Access key" required autofocus>
  <button>Continue</button>
</form></body></html>"""


def is_local_request():
    return request.remote_addr in ("127.0.0.1", "::1")


def require_access_key(app, key):
    """Require `key` on every request from another device (None turns it off)."""
    app.config["ACCESS_KEY"] = key
    if app.config.get("_access_key_hooks"):
        return
    app.config["_access_key_hooks"] = True

    @app.before_request
    def _check_access_key():
        expected = app.config.get("ACCESS_KEY")
        if not expected or is_local_request():
            return None
        supplied = (request.args.get("key") or request.headers.get("X-Access-Key")
                    or request.cookies.get(COOKIE_NAME) or "")
        if hmac.compare_digest(supplied.encode(), expected.encode()):
            g.remember_access_key = "key" in request.args
            return None
        if request.path.startswith("/api/") or request.path == "/video_feed":
            return jsonify({"error": "access key required"}), 401
        return Response(LOCKED_PAGE, status=401, mimetype="text/html")

    @app.after_request
    def _remember_access_key(response):
        if g.get("remember_access_key"):
            response.set_cookie(COOKIE_NAME, app.config["ACCESS_KEY"], max_age=COOKIE_MAX_AGE,
                                httponly=True, samesite="Strict", secure=request.is_secure)
        return response
