# Safe Code Editor

Use this skill when the task requires changing code, tests, docs, or project
configuration.

Working rules:

- Search before reading broad files.
- Read only the files needed for the current change.
- Prefer `apply_patch` for edits.
- Do not rewrite unrelated code.
- Preserve user changes.
- Run focused tests or syntax checks after changes.
- Report what changed and what was verified.

Expected output:

- Changed files
- Behavior change
- Tests or checks run
- Remaining risk
