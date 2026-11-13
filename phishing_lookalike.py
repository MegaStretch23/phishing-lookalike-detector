# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "playwright>=1.45",
#   "opencv-python-headless>=4.9",
#   "imagehash>=4.3",
#   "pillow>=10",
#   "numpy>=1.26",
# ]
# ///
"""Screenshot-based phishing lookalike detector.

Pipeline: headless Playwright screenshot -> perceptual hash (pHash) of the whole page +
ORB keypoints/RANSAC inliers on the header (logo) band -> best-matching reference brand.
If a page looks like brand X but its registrable domain is not one of X's domains, the
verdict is "IMPERSONATES X".

Tonight's corpus is offline: brand login pages are synthetic stand-ins generated below
and served to the real browser through Playwright request interception (context.route),
so goto("https://paypa1-verify.com/...") renders a local fixture - no socket, no port,
no outbound request (anything not in the fixture map is aborted).

  uv run main.py                 build refs, run the 20-URL demo set, print verdict table
  uv run main.py --selfcheck     same, then re-read out/verdicts.json + PNGs and assert
  uv run main.py --live URL ...  score real URLs against the refs (network; opt-in)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

import cv2
import imagehash
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
REFS = OUT / "refs"
SHOTS = OUT / "shots"
VERDICTS = OUT / "verdicts.json"
VIEWPORT = {"width": 1280, "height": 800}
HEADER_FRAC = 0.45  # top band that carries the logo / brand chrome

# Match thresholds (calibrated on the demo set; see README).
MIN_INLIERS = 60  # cross-brand runner-ups peak ~38, true matches >=200
MIN_PHASH_SIM = 0.62

# ---------------------------------------------------------------- brand fixtures
# name, legit registrable domains, bg, accent, logo-shape svg, wordmark
BRANDS = [
    ("PayPal", ["paypal.com"], "#f5f7fa", "#003087", "circle", "PayPal"),
    ("Microsoft", ["microsoft.com", "live.com", "microsoftonline.com"], "#ffffff", "#0067b8", "grid", "Microsoft"),
    ("Chase", ["chase.com"], "#e8eef5", "#117aca", "octagon", "CHASE"),
    ("BankOfAmerica", ["bankofamerica.com"], "#ffffff", "#e31837", "flag", "BANK OF AMERICA"),
    ("WellsFargo", ["wellsfargo.com"], "#d71e28", "#ffcd41", "stagecoach", "WELLS FARGO"),
    ("Apple", ["apple.com", "icloud.com"], "#fbfbfd", "#1d1d1f", "apple", "Apple ID"),
    ("Google", ["google.com"], "#ffffff", "#4285f4", "ring", "Google"),
    ("Amazon", ["amazon.com"], "#ffffff", "#ff9900", "smile", "amazon"),
    ("Netflix", ["netflix.com"], "#141414", "#e50914", "bars", "NETFLIX"),
    ("Coinbase", ["coinbase.com"], "#ffffff", "#0052ff", "cmark", "coinbase"),
]


def _logo_svg(shape: str, accent: str) -> str:
    s = {
        "circle": f'<circle cx="40" cy="40" r="32" fill="{accent}"/><circle cx="40" cy="40" r="14" fill="#fff"/><rect x="30" y="30" width="8" height="30" fill="{accent}"/>',
        "grid": '<rect x="4" y="4" width="34" height="34" fill="#f25022"/><rect x="42" y="4" width="34" height="34" fill="#7fba00"/><rect x="4" y="42" width="34" height="34" fill="#00a4ef"/><rect x="42" y="42" width="34" height="34" fill="#ffb900"/>',
        "octagon": f'<polygon points="24,4 56,4 76,24 76,56 56,76 24,76 4,56 4,24" fill="{accent}"/><rect x="28" y="28" width="24" height="24" fill="#fff"/>',
        "flag": f'<path d="M4 20 L76 8 L76 22 L4 34Z" fill="{accent}"/><path d="M4 40 L76 28 L76 42 L4 54Z" fill="#012169"/><path d="M4 60 L76 48 L76 62 L4 74Z" fill="{accent}"/>',
        "stagecoach": f'<rect x="6" y="16" width="68" height="40" rx="6" fill="{accent}"/><circle cx="22" cy="62" r="12" fill="#222"/><circle cx="58" cy="62" r="12" fill="#222"/><rect x="16" y="24" width="18" height="14" fill="#fff"/><rect x="44" y="24" width="18" height="14" fill="#fff"/>',
        "apple": f'<ellipse cx="40" cy="48" rx="28" ry="26" fill="{accent}"/><ellipse cx="50" cy="14" rx="7" ry="12" transform="rotate(35 50 14)" fill="{accent}"/><circle cx="66" cy="40" r="10" fill="#fbfbfd"/>',
        "ring": '<path d="M40 8 A32 32 0 0 1 72 40 L56 40 A16 16 0 0 0 40 24Z" fill="#ea4335"/><path d="M72 40 A32 32 0 0 1 40 72 L40 56 A16 16 0 0 0 56 40Z" fill="#34a853"/><path d="M40 72 A32 32 0 0 1 8 40 L24 40 A16 16 0 0 0 40 56Z" fill="#fbbc05"/><path d="M8 40 A32 32 0 0 1 40 8 L40 24 A16 16 0 0 0 24 40Z" fill="#4285f4"/>',
        "smile": f'<path d="M8 44 Q40 76 72 44 L68 40 Q40 64 12 40Z" fill="{accent}"/><polygon points="64,34 76,44 62,48" fill="{accent}"/><rect x="18" y="8" width="44" height="24" rx="4" fill="#232f3e"/>',
        "bars": f'<rect x="10" y="4" width="16" height="72" fill="{accent}"/><rect x="54" y="4" width="16" height="72" fill="{accent}"/><polygon points="10,4 26,4 70,76 54,76" fill="#b20710"/>',
        "cmark": f'<circle cx="40" cy="40" r="34" fill="{accent}"/><circle cx="40" cy="40" r="18" fill="#fff"/><rect x="44" y="30" width="32" height="20" fill="{accent}"/>',
    }[shape]
    return f'<svg width="80" height="80" viewBox="0 0 80 80">{s}</svg>'


def brand_page(brand: tuple, variant: str = "legit") -> str:
    """variant: legit | phish (a kit copy: reworded copy, urgency banner, shifted card)."""
    name, _, bg, accent, shape, word = brand
    dark = bg in ("#141414", "#d71e28")
    fg = "#fff" if dark else "#1a1a1a"
    shift = "margin-top:34px" if variant == "phish" else "margin-top:60px"
    banner = ('<div style="background:#fff3cd;color:#664d03;padding:8px;font:14px Arial;text-align:center">'
              'Unusual sign-in activity detected. Verify your account within 24 hours.</div>') if variant == "phish" else ""
    heading = "Confirm your identity" if variant == "phish" else "Sign in"
    btn = "Verify now" if variant == "phish" else "Log In"
    return f"""<!doctype html><html><head><title>{name}</title></head>
<body style="margin:0;background:{bg};font-family:Arial,Helvetica,sans-serif;color:{fg}">{banner}
<div style="height:90px;border-bottom:4px solid {accent};display:flex;align-items:center;padding-left:48px;gap:16px">
{_logo_svg(shape, accent)}<span style="font:bold 34px Georgia,serif;letter-spacing:1px;color:{accent if not dark else '#fff'}">{word}</span></div>
<div style="width:380px;margin-left:auto;margin-right:auto;{shift};padding:32px;border:1px solid #ccc;border-radius:10px;background:{'#222' if dark else '#fff'}">
<div style="text-align:center">{_logo_svg(shape, accent)}</div>
<h2 style="text-align:center;font-weight:normal">{heading}</h2>
<input placeholder="Email or username" style="width:100%;padding:12px;margin:8px 0;box-sizing:border-box">
<input type="password" placeholder="Password" style="width:100%;padding:12px;margin:8px 0;box-sizing:border-box">
<button style="width:100%;padding:12px;background:{accent};color:#fff;border:0;border-radius:6px;font-size:16px">{btn}</button>
<p style="font-size:12px;text-align:center">Forgot password? &middot; Create account</p></div></body></html>"""


def generic_login() -> str:
    return """<!doctype html><html><body style="margin:0;background:#eef;font-family:Verdana">
<div style="width:340px;margin:140px auto;padding:28px;background:#fff;border-radius:4px">
<h3>Member login</h3><input placeholder="user" style="width:100%;padding:10px;margin:6px 0">
<input type="password" placeholder="pass" style="width:100%;padding:10px;margin:6px 0">
<button style="width:100%;padding:10px">Submit</button></div></body></html>"""


def blog_page(title: str, color: str) -> str:
    paras = "".join(f"<p>Paragraph {i}: notes on {title.lower()} — lorem ipsum dolor sit amet, consectetur "
                    f"adipiscing elit, sed do eiusmod tempor incididunt ut labore.</p>" for i in range(8))
    return f"""<!doctype html><html><body style="margin:0;font-family:'Times New Roman';background:#fffdf7">
<div style="background:{color};color:#fff;padding:18px 40px;font-size:26px">{title}</div>
<div style="max-width:760px;margin:30px auto;font-size:17px;line-height:1.6">{paras}</div></body></html>"""


def build_corpus() -> tuple[dict[str, str], list[dict]]:
    """Returns (url -> html for routing, test cases with expected verdict)."""
    pages: dict[str, str] = {}
    for b in BRANDS:
        pages[f"https://ref.local/{b[0]}"] = brand_page(b, "legit")
    by = {b[0]: b for b in BRANDS}
    phish = [  # (url, brand) - a newly-registered lookalike domain serving a kit copy
        ("https://paypa1-account-verify.com/signin", "PayPal"),
        ("https://login-microsoftonline.support/common/oauth2", "Microsoft"),
        ("https://chase-secure-alerts.net/logon", "Chase"),
        ("https://bofa-online-verify.info/login", "BankOfAmerica"),
        ("https://wellsfargo.account-review.co/signon", "WellsFargo"),
        ("https://appleid.apple.com-locked.xyz/", "Apple"),
        ("https://accounts-google.security-check.top/", "Google"),
        ("https://amazon-billing-update.shop/ap/signin", "Amazon"),
        ("https://netflix-payment-hold.com/login", "Netflix"),
        ("https://coinbase-wallet-restore.app/signin", "Coinbase"),
    ]
    legit = [
        ("https://www.paypal.com/signin", "PayPal"),
        ("https://login.microsoftonline.com/common/oauth2", "Microsoft"),
        ("https://secure.chase.com/logon", "Chase"),
        ("https://www.netflix.com/login", "Netflix"),
    ]
    cases = []
    for url, name in phish:
        pages[url] = brand_page(by[name], "phish")
        cases.append({"url": url, "expect": f"IMPERSONATES {name}"})
    for url, name in legit:
        pages[url] = brand_page(by[name], "legit")
        cases.append({"url": url, "expect": f"LEGIT {name}"})
    benign = [
        ("https://intranet-portal.example.org/login", generic_login()),
        ("https://gardening-weekly.blog/tomatoes", blog_page("Growing Tomatoes", "#2e7d32")),
        ("https://localnews.example.net/today", blog_page("Local News Today", "#37474f")),
        ("https://recipes.example.com/bread", blog_page("Sourdough Basics", "#8d6e63")),
        ("https://tech-notes.dev/rust", blog_page("Rust Ownership Notes", "#b7410e")),
        ("https://club-members.example.com/login", generic_login().replace("#eef", "#fee").replace("Member", "Club")),
    ]
    for url, html in benign:
        pages[url] = html
        cases.append({"url": url, "expect": "NO_MATCH"})
    return pages, cases


# ---------------------------------------------------------------- capture
def slug(url: str) -> str:
    p = urlparse(url)
    return (p.hostname or "x").replace(".", "_") + "_" + hashlib.sha1(url.encode()).hexdigest()[:6]


def screenshot_all(targets: list[tuple[str, Path]], pages: dict[str, str] | None) -> None:
    """Screenshot each (url, png). If `pages` is given, every request is served from it
    via route interception and anything else is aborted (fully offline)."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = None
        try:
            try:
                browser = pw.chromium.launch(headless=True)
            except Exception:
                browser = pw.chromium.launch(headless=True, channel="msedge")
            ctx = browser.new_context(viewport=VIEWPORT, device_scale_factor=1)
            if pages is not None:
                def handle(route):
                    html = pages.get(route.request.url)
                    if html is None:
                        route.abort()
                    else:
                        route.fulfill(status=200, content_type="text/html", body=html)
                ctx.route("**/*", handle)
            page = ctx.new_page()
            for url, png in targets:
                png.parent.mkdir(parents=True, exist_ok=True)
                page.goto(url, wait_until="load", timeout=30000)
                page.screenshot(path=str(png), full_page=False)
        finally:
            if browser is not None:
                browser.close()


# ---------------------------------------------------------------- features / matching
_orb = cv2.ORB_create(nfeatures=1500)


def features(png: Path) -> dict:
    img = Image.open(png).convert("RGB")
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    band = gray[: int(gray.shape[0] * HEADER_FRAC)]
    kps, des = _orb.detectAndCompute(band, None)
    return {"phash": imagehash.phash(img, hash_size=8), "kps": kps, "des": des}


def orb_inliers(a: dict, b: dict) -> int:
    if a["des"] is None or b["des"] is None or len(a["kps"]) < 8 or len(b["kps"]) < 8:
        return 0
    m = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(a["des"], b["des"], k=2)
    good = [p[0] for p in m if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < 8:
        return 0
    src = np.float32([a["kps"][g.queryIdx].pt for g in good]).reshape(-1, 1, 2)
    dst = np.float32([b["kps"][g.trainIdx].pt for g in good]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return int(mask.sum()) if mask is not None else 0


def registrable(host: str) -> str:
    parts = host.lower().strip(".").split(".")
    if len(parts) >= 3 and parts[-2] in ("co", "com", "org", "net", "ac", "gov") and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def judge(url: str, feat: dict, refs: dict[str, dict]) -> dict:
    scored = []
    for name, rf in refs.items():
        sim = 1 - (feat["phash"] - rf["phash"]) / 64
        scored.append((orb_inliers(feat, rf), round(sim, 3), name))
    scored.sort(reverse=True)
    inl, sim, name = scored[0]
    dom = registrable(urlparse(url).hostname or "")
    legit = {d for b in BRANDS if b[0] == name for d in b[1]}
    if inl >= MIN_INLIERS and sim >= MIN_PHASH_SIM:
        verdict = f"LEGIT {name}" if dom in legit else f"IMPERSONATES {name}"
    else:
        verdict = "NO_MATCH"
    return {"url": url, "domain": dom, "best_brand": name, "orb_inliers": inl, "phash_sim": sim,
            "runner_up_inliers": scored[1][0], "verdict": verdict}


def build_refs(pages: dict[str, str]) -> dict[str, dict]:
    targets = [(f"https://ref.local/{b[0]}", REFS / f"{b[0]}.png") for b in BRANDS]
    screenshot_all(targets, pages)
    return {b[0]: features(REFS / f"{b[0]}.png") for b in BRANDS}


def print_table(rows: list[dict]) -> None:
    print(f"{'url':52} {'best':14} {'inl':>4} {'phash':>5}  verdict")
    for r in rows:
        mark = "" if r.get("expect") in (None, r["verdict"]) else f"   <-- expected {r['expect']}"
        print(f"{r['url'][:52]:52} {r['best_brand']:14} {r['orb_inliers']:>4} {r['phash_sim']:>5}  {r['verdict']}{mark}")


def run_demo() -> list[dict]:
    pages, cases = build_corpus()
    refs = build_refs(pages)
    targets = [(c["url"], SHOTS / f"{slug(c['url'])}.png") for c in cases]
    screenshot_all(targets, pages)
    rows = []
    for c, (url, png) in zip(cases, targets):
        r = judge(url, features(png), refs)
        r.update(expect=c["expect"], screenshot=str(png.relative_to(HERE)).replace("\\", "/"),
                 sha256=hashlib.sha256(png.read_bytes()).hexdigest())
        rows.append(r)
    VERDICTS.write_text(json.dumps({"thresholds": {"min_inliers": MIN_INLIERS, "min_phash_sim": MIN_PHASH_SIM},
                                    "refs": [b[0] for b in BRANDS], "results": rows}, indent=2), encoding="utf-8")
    print_table(rows)
    ok = sum(r["verdict"] == r["expect"] for r in rows)
    print(f"\n{ok}/{len(rows)} verdicts as expected -> {VERDICTS.relative_to(HERE)}")
    return rows


def selfcheck() -> None:
    run_demo()
    data = json.loads(VERDICTS.read_text(encoding="utf-8"))  # re-read the artifact
    rows = data["results"]
    assert len(rows) == 20, f"expected 20 results, got {len(rows)}"
    for b in data["refs"]:
        assert (REFS / f"{b}.png").stat().st_size > 5000, f"ref screenshot missing/empty: {b}"
    for r in rows:
        png = HERE / r["screenshot"]
        assert png.exists(), png
        assert hashlib.sha256(png.read_bytes()).hexdigest() == r["sha256"], f"screenshot changed: {png}"
        w, h = Image.open(png).size
        assert (w, h) == (VIEWPORT["width"], VIEWPORT["height"]), (png, w, h)
        assert r["verdict"] == r["expect"], f"{r['url']}: got {r['verdict']}, expected {r['expect']}"
    imps = [r for r in rows if r["verdict"].startswith("IMPERSONATES")]
    assert len(imps) == 10 and len({r["best_brand"] for r in imps}) == 10, "each phish must map to its own brand"
    for r in imps:  # the impersonated brand must win clearly, not by a hair
        assert r["orb_inliers"] >= 2 * max(r["runner_up_inliers"], 1), r
    assert all(r["verdict"] == "NO_MATCH" for r in rows if "example" in r["url"] or r["url"].endswith(("tomatoes", "rust")))
    # the verdict is domain-driven: same pixels at the real domain are LEGIT
    assert {r["verdict"] for r in rows if r["domain"] in ("paypal.com", "chase.com")} == {"LEGIT PayPal", "LEGIT Chase"}
    print("SELFCHECK OK: 20/20 verdicts match; screenshots re-read and hash-verified")


def run_live(urls: list[str]) -> None:
    pages, _ = build_corpus()
    refs = build_refs(pages)
    targets = [(u, SHOTS / "live" / f"{slug(u)}.png") for u in urls]
    screenshot_all(targets, None)
    print_table([judge(u, features(p), refs) for u, p in targets])


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--live", nargs="+", metavar="URL", help="score real URLs (hits the network)")
    a = ap.parse_args()
    if a.selfcheck:
        selfcheck()
    elif a.live:
        run_live(a.live)
    else:
        run_demo()


if __name__ == "__main__":
    main()
