# phishing-lookalike-detector

Screenshot-based lookalike-page triage in Python. A headless Chromium (Playwright) screenshot of a login page is compared with brand reference screenshots by perceptual hash (pHash) plus ORB logo keypoints with a RANSAC homography. If the page looks like brand X but sits on a domain X does not own, the verdict is `IMPERSONATES X`. The same pixels on a domain X owns come back `LEGIT X`, so the verdict depends on the domain, not just the picture.

MIT licensed, single file (`phishing_lookalike.py`).

## Quickstart (about 15 seconds after dependencies are installed)

Needs Python 3.11+. Chromium is only needed for Playwright (`playwright install chromium`; the script falls back to Microsoft Edge if present).

```bash
git clone https://github.com/MegaStretch23/phishing-lookalike-detector.git
cd phishing-lookalike-detector

# option A: uv (reads the dependency header inside the script)
uv run phishing_lookalike.py --selfcheck

# option B: plain pip
pip install -r requirements.txt && playwright install chromium
python phishing_lookalike.py --selfcheck
```

The demo is fully offline: every request goes through Playwright `context.route` and is served from an in-memory fixture map, anything else is aborted. No port is bound and nothing leaves your machine. It writes `out/refs/*.png`, `out/shots/*.png` and `out/verdicts.json`.

`--selfcheck` runs the pipeline, then re-reads `out/verdicts.json` and asserts all 20 verdicts, that every screenshot is 1280x800 and matches its recorded sha256, and that the same pixels served on `paypal.com` / `chase.com` come back LEGIT.

Score real URLs (this one uses the network):

```bash
uv run phishing_lookalike.py --live https://example.com/login
```

## Real saved output

This is the unedited result of a run on 2026-11-13 (Linux, headless Chromium, 14 s wall time). Full file: [`examples/sample-run-2026-11-13.txt`](examples/sample-run-2026-11-13.txt), per-URL numbers and screenshot hashes: [`examples/sample-verdicts-2026-11-13.json`](examples/sample-verdicts-2026-11-13.json).

```
url                                                  best            inl phash  verdict
https://paypa1-account-verify.com/signin             PayPal          286 0.844  IMPERSONATES PayPal
https://login-microsoftonline.support/common/oauth2  Microsoft       350 0.875  IMPERSONATES Microsoft
https://chase-secure-alerts.net/logon                Chase           183 0.844  IMPERSONATES Chase
https://www.paypal.com/signin                        PayPal          539   1.0  LEGIT PayPal
https://secure.chase.com/logon                       Chase           419   1.0  LEGIT Chase
https://gardening-weekly.blog/tomatoes               Chase            23   0.5  NO_MATCH
https://club-members.example.com/login               Coinbase         11 0.438  NO_MATCH
... (20 rows in the full file)

20/20 verdicts as expected -> out/verdicts.json
SELFCHECK OK: 20/20 verdicts match; screenshots re-read and hash-verified
```

On the demo set: 10 lookalikes caught (183 to 434 logo keypoint inliers, best wrong-brand runner-up 32), 4 real-domain copies LEGIT, 6 unrelated pages NO_MATCH (at most 31 inliers).

## Honest limits

- The 10 brands are synthetic stand-ins (inline SVG logo plus wordmark), not real brand pages, and the lookalikes are built from the same template as the references. This is an easy case. It has not been tested on real phishing pages; the thresholds (60 inliers, 0.62 pHash similarity) were set on this demo set and need re-tuning on yours.
- `--live` has no per-URL error handling, no redirect or iframe handling, and no wait for JS-heavy pages beyond `load`.
- `registrable()` is a crude eTLD+1 heuristic; use `tldextract` or the Public Suffix List for real work.
- ORB copes poorly with recolored or vector-text logos. This is a starter to learn from and build on, not a production detector.

## Want the packaged kit?

The paid source kit adds a buyer-oriented package of the same pipeline: **[Phishing Lookalike Detector source kit on Gumroad, $29](https://vedovino.gumroad.com/l/phishing-lookalike-detector)**. The code in this repo is free; the kit is for people who would rather buy it assembled.
