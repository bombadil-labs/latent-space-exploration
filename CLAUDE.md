# latent-space-exploration

Activation-level toolkit for measuring narrative/relational structure in transformer residual
streams. Honest running record in `RESULTS.md` (Checkpoint 2 at the top says what stands, what fell
and what was **withdrawn**). Claims in `WRITEUP.md`. Calculus in `docs/ALGEBRA.md`.

## Read these before touching measurement code

- **`docs/INSTRUMENTS.md`** — the six instruments this project found broken and what each invalidated.
- **`docs/DELEGATION.md`** — the plan → review → execute → screen → integrate flow, and the rules each
  retraction bought.
- **`docs/specs/core_v1.md`** — the verified measurement core. Not built yet.

## Non-negotiables

1. **Every battery reports treatment, random, AND no-patch.** A battery without all three is not a
   result. Every arm declares where it should sit; an arm off its null is a bug until proven otherwise.
2. **Any readout at or after a patch layer needs a pass-through arm** (the offline arithmetic:
   `readout(base_resid + shift)`, no forward pass). Stage 14 was withdrawn for want of this.
3. **Estimate the noise floor before believing a null.** Three negatives were withdrawn as an
   instrument reading itself.
4. **Never select a layer on scoring data.** Report the curve.
5. **Read the diff of agent code touching extraction, patching or ranking — not the report.** Two of
   six bugs arrived via a summary I trusted.
6. **Never `pgrep -f`/`pkill -f` a pattern contained in your own command string.** Cost so far: one
   false kill and 48 wasted minutes.

## Environment

- `.venv` — py3.11, CPU torch, local models. `.venv312` — py3.12, nnsight, NDIF.
- `HF_HOME=$PWD/cache/hf`, `HF_HUB_DISABLE_XET=1`, `HF_HUB_OFFLINE=1` for local runs.
- Llama-3.1-405B is **not** reachable with this key (h34). NDIF and TypeSafe credentials come from
  the environment/proxy; never print them.
- Disk is limited. Download nothing large. `.npz` stacks are gitignored and will not survive.

## Conventions

Develop on `claude/amazing-faraday-881p04`. Do not open pull requests. Record negatives and confounds
in `RESULTS.md` with the same care as positives; retraction is a first-class operation here.
