# Verification record

Run `node web/verify.cjs` from the repository root. The checks evaluate the exact JavaScript embedded in `web/index.html`, with no network dependencies. Thirteen deterministic checks cover IPv4 boundaries (/0,/24,/31,/32), malformed addresses, known linear-system solutions, pivoting, singular/malformed matrices, booking thresholds, invalid observations, time-series baseline calculations and demand-filter aggregates.

All 13 checks passed on 14 November 2026. HTML document structure and private-path/student-number absence were also checked. Browser calculations use synthetic inputs; these checks do not validate the original assessed application, certify security of original PHP/Python code, establish model performance, or prove a live deployment.
