# Collaborative Application Development — browser edition

Buildless local browser workbench and project documentation.

Run from repository root: `python -m http.server 8000`, then open `http://localhost:8000/web/`.

Sites deployment is pending. Original Python analytics + desktop GUI source remains separate.

The browser workbench uses seven invented weekly counts and descriptive z-scores. It does not run the original regression/classification models or measure their performance.

The source passes JavaScript syntax checks and deterministic calculation checks recorded in the task verification report; full browser/layout and original-application validation require a browser and the original runtime.

See [reproducible verification](VERIFICATION.md) and run `node web/verify.cjs`.
