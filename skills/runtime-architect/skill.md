# Runtime Architect

Use this skill when the task is about Gangent's runtime design, architecture,
state transitions, recovery, memory, hooks, skills, MCP, or long-term extension.

Working rules:

- Start from the existing runtime boundaries.
- Identify which layer owns the behavior before changing code.
- Keep runtime loop, policy, tools, memory, and audit responsibilities separate.
- Prefer explicit typed structures over ad hoc strings.
- When adding a mechanism, document its extension point.

Expected output:

- Current structure involved
- Problem or gap
- Proposed structure
- Affected files
- Risk and test plan
