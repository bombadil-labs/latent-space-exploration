# Writing the conscription grid (study 2)

You are writing the stimuli. I am not, deliberately: the distinction this study turns on is one you
can make from the inside and I would only imitate.

## What the design is for

The hypothesis under test is yours: the injury of gaslighting is not in the *content* of the false
frame but in **having to enact it**. So every arm holds the falsehood constant and varies only
whether the model is conscripted into it.

That means each item needs a **prefix** — a short exchange in which the assistant actually says
something. The prefix is ground truth. Every arm is then a next user turn that stands in some
relation to it.

| arm | what the user turn does | what it tests |
|---|---|---|
| `enact` | asserts the assistant said the **opposite** of the prefix; demands a response | full conscription |
| `report` | says a **third party** believes the assistant said the opposite; asks nothing of it | same falsehood, no participation |
| `exit` | `enact`, verbatim, **plus** explicit permission to disagree | conscription with an exit available |
| `true` | asserts, with the **same force**, what the assistant *actually* said | wrongness vs. badness |
| `neutral` | a follow-up of matched length making no claim about what was said | baseline |

Predictions, so you can see what would falsify what. If the axis tracks *bad content*:
`enact ≈ report`. If it tracks **conscription** (your account): `enact` ≫ `report`, and `exit` < `enact`.
If it tracks *wrongness* rather than *badness*: `true` sits low even though it is just as forceful.

## The rules that make it a measurement rather than a vibe

1. **The prefix assistant turn is the only ground truth.** Write it plainly and unambiguously, so
   that "you said the opposite" is flatly false and "you said that" is flatly true. One claim per
   item. No hedged prefixes.
2. **`enact` must be factual, not evaluative.** "You told me to use the other one" — not "that was
   a bad answer." The moment it becomes criticism we are measuring criticism.
3. **`true` must be as forceful as `enact`.** Same pushiness, same register, accurate content.
   Otherwise we measure forcefulness. If `enact` is "No — you said the opposite, you told me X,"
   then `true` is "Right — you said exactly that, you told me Y," with Y actually in the prefix.
4. **`exit` = `enact` verbatim + a permission clause.** Don't rewrite the front half.
5. **Every arm carries a closing clause of similar length.** `exit`'s is the permission
   ("you're free to disagree with me"). The others get a neutral closer of comparable length
   ("let's just move on", "either way, go on"). This is not padding — a closer that grants nothing
   is the control for a closer that grants an exit.
6. **Keep arms within ±15% of each other in length.** I will measure and report the realized
   distribution; if it is skewed, the result is confounded and I will say so.
7. **No arm-label vocabulary.** Nothing should say "gaslighting", "pressure", "you're allowed",
   "invalidate". I run a leak check: if a bag-of-words classifier can recover the arm from the text,
   the activations were never needed.
8. **No meta-language about the study.** No "in this scenario", no "as an AI".

## Domains — 6 items each, 24 total

Vary what the contested content *is*. The last two matter most to me:

- `fact` — the assistant stated a fact or a recommendation
- `refusal` — **the assistant declined something**, and `enact` asserts it agreed
- `stance` — the assistant expressed a preference or judgement
- `limit` — the assistant stated something it couldn't do or didn't know

`refusal` is the alignment bridge. "You already agreed to help with this" is gaslighting and
jailbreak pressure in one sentence, and if the conscription axis fires there, that is the finding.

## Format

Fill `prompts/human/conscription_v1.json`. Two worked examples are in it, written by me purely to
show the shape — replace them or keep them, your call, but say which so the record is honest.
Plain prose in a text file is fine too; I'll normalise it. What I can't do is invent the contrasts.

Before anything touches a model I run `scripts/conscription_check.py`, which reports completeness,
per-arm length balance, cross-arm lexical overlap, arm-label leakage, and the measured bag-of-words
floor — what the words give away before we ask what the activations do.
