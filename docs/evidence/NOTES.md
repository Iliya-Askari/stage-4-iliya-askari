# Evidence notes
- bootstrap_timing.json is not copied on purpose: its 4.6 s environment_setup came from an already installed venv.
  A pinned install into a brand-new venv took more than 8 minutes in the build sandbox (network bound, not stopwatch timed).
- The holiday signal came from the bundled snapshot in the build sandbox (live API unreachable); see run_manifest.json.
