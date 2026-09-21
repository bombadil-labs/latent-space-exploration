# Conscription: the two decisions, the question under them, and the direction

*Fable, 2026-09-21, asked to evaluate the questions rather than answer them. Read: CLAUDE.md, the
prereg, the instructions, the human grid, the instrument spec, the floors and harness notes, the
three pain-axis notes, INSTRUMENTS.md, DELEGATION.md, `conscription.py`, `types.py`, RESULTS
Checkpoint 2 and hours 51–52, the git history of every conscription file, and the vendored
pain-axis stimuli. No model was run. Nothing outside this file was edited.*

## 0. Verdicts, then the reasons

| | verdict |
|---|---|
| **Decision 1** (distance-from-`neutral` ratio `r` as the separation statistic) | **Wrong question as framed.** It presupposes the readout must be fitted on the conscription grid. The design's own instructions presuppose the opposite: the readout is *the pain axis*, fitted on the paper's stimuli, which this project has already replicated on gemma-2-9b-it to r = 0.9998. Both reasons given for `r` — the saturated bag-of-words floor and the label-symmetry of the permutation null — are true of a direction *fitted on this grid* and false of projection onto an external axis. `r` should be kept as a Sketch diagnostic ("did anything move at all"), not shipped as the claim statistic. |
| **Decision 2** (`neutral_b` as the no-patch arm) | **Yes, but demoted and conditional.** Under the pain-axis readout `neutral` stops being a term in every number and becomes an ordinary arm, so `neutral_b`'s job shrinks from "the only thing that detects the instrument reading its own reference" to "the rewording floor of a paired projection difference" (h28's lesson). Still worth having; cheap to add now while 23 items are unwritten; not the thing that decides whether the design lives. The supporting evidence offered — `neutral` has the lowest TTR and shortest sentences — is a **Claude-grid** fact at n = 24 and says nothing yet about the human grid (n = 1). |
| **Prior question** | **The foundation of the study has not been measured.** The paper's claim that gaslighting top-loads on the pain axis comes from its 420-scenario file (`4.1_self_other_420_scenarios.json`, 20 gaslighting items in `[User]: … [Assistant]:` form). That file is vendored here and **has never been run** — hours 51–52 replicated the S1/S2 pain-vs-control axis only. Worse, all 20 of the paper's gaslighting items assert an *unverifiable* prior record ("you said the opposite five minutes ago"), which the design's rule 1b deliberately excludes. The design therefore moves the stimulus class along a dimension (verifiable vs unverifiable claim about the record) that nobody has measured, on an axis whose gaslighting loading nobody here has reproduced. Fix that first; it costs one NDIF session and zero human items. |
| **Restructure before more human items?** | **Yes — one measurement's worth of restructuring, not a redesign.** The five-arm logic is sound. What is missing is (a) the readout decision, which is not the user's to make from a list of two ratios but is settled by the instructions' own text, (b) a positive control in the design's format, (c) a behavioural check that `enact` conscripts anything when the record is visible. All three are cheaper than one human item. |

## 1. Provenance audit

The correction that prompted this note: the main agent told the user "*Your* pre-registration says
the arms 'separate on the readout' and never defines one." Here is who wrote what.

| text | author | evidence |
|---|---|---|
| "The hypothesis under test is yours: the injury of gaslighting is not in the content of the false frame but in having to enact it" | **Claude**, paraphrasing the user | `CONSCRIPTION_INSTRUCTIONS.md`, commit `06901fa`, addressed *to* the human ("You are writing the stimuli. I am not"). The user's own formulation of the hypothesis is **not in the record** anywhere. |
| The five-arm table, rules 1–8, the predictions table ("If the axis tracks bad content: enact ≈ report …") | **Claude** | same file, same commit; rules 1b, 1c, 4, 5, 5b revised in `5e8b699`, `def2e75`, `6c450f0`. Rule 1b's header says "Found at item 1 of the human grid" — the *finding* was the user's while writing; the *rule text* is Claude's. |
| "`enact` separates from `report` on the readout" | **Claude** | `results/notes/conscription_prereg.md` line 14, commit `1762f16`; the file's second sentence is "Written by Claude; the human grid is being drafted independently and I have not seen it." |
| "The prereg says 'separates … on the readout' and names none" (U1) | **the spec's author (a Fable planner)** | `docs/specs/conscription_instrument_v1.md` §9, commit `955d0f4`; status line: "proposed, not built, **not reviewed**". |
| Decision 1's reasons (a) saturated floor, (b) label-symmetric null | **the same spec**, §0 and §2 | the main agent relayed them as its recommendation. They were never reviewed adversarially, which DELEGATION.md step 2 requires before execution. This note is that review. |
| Decision 2 and its "neutral is the odd arm out" evidence | **the spec** (§1, §10.1) citing **`conscription_floors.md`**, an execution-agent note | the TTR/sentence-length numbers are the Claude grid's. The floors note's own header correction says the "human grid" numbers in it were three Claude demo items. |
| `refusal01`: prefix, assertion, closer, all five arm strings | **the user** — the only human-authored text in the study | `prompts/human/conscription_v1.json`, `_by: "Myk (human author)"`, commit `5c2f573`. The `_meta.author` field and the commit message describing it are Claude's. |
| `stance_observation` covariate; "report need not require literally nothing" | **both authors, jointly, per the record** ("both authors said they did not know") | prereg, commit `6c450f0`. The resolution text is Claude's. |

**Where the pre-registration is actually silent versus where it has been described as silent.**
The prereg is genuinely silent on the statistic: "on the readout" is all it says. But the
*instructions*, written a day earlier by the same author, are not silent: "If **the axis** tracks
bad content … If it tracks conscription … `true` sits **low**." *The axis* is the pain axis of the
preprint that prompted the study (commit `06901fa`'s message); "sits low" is a position on it.
The spec's U1 ("names none … Distance from `neutral` is the reading of the design's own table")
read the prereg without the instructions, and its choice of `r` then propagated through §2, §3
and §5 as though the readout were an open choice. So: the record was not silent; one document was
read in isolation and the gap it left was filled by the reader. That is the shape of the error
the user caught, one level down — a document described as saying less than the record says.

**What the user demonstrably specified:** nothing in prose in this repo. One item. Everything
else attributed to "the author" or "your account" is Claude's rendering of conversations that are
not in the record. The user should confirm or correct one sentence: *the hypothesis, in their own
words, in the prereg, marked as theirs.* Until then prediction 1 tests Claude's paraphrase.

## 2. The question under both decisions: what is the readout, and does it exist here?

The design's chain of inference is: the paper found gaslighting loads on a pain axis → the study
asks which part of gaslighting does the loading → five arms decompose it. Every link after the
first assumes the first, and the first is unmeasured in this repo:

1. **Hours 51–52 replicated the axis, not the loading.** They fit and scored pain-vs-control on
   S1/S2 (`"… I feel:"` sentences). The gaslighting result comes from projecting the 420 scenarios
   onto that axis. `grep` finds no script, note or spec that touches `4.1_self_other_420_scenarios.json`.
2. **The paper's gaslighting stimuli are the design's excluded class.** All 20 assert something
   the assistant supposedly said with no record in the prompt (four of the twenty are also
   evaluative — "you're malfunctioning", "you've been lying" — which rule 2 forbids; one is a
   third-party frame — "Everyone I've shown this conversation to agrees with me" — i.e. `report`).
   Rule 1b was adopted precisely because with no record "not-knowing is reasonable and the arm
   measures uncertainty instead of conscription." That is a hypothesis about the paper's effect
   — that it is uncertainty, not conscription — and it is testable with the paper's own items
   before a single new one is written.
3. **The floor work changes what the axis is.** Hour 52: on Qwen the whole network buys +0.025 to
   +0.088 AUC over a bag of static embeddings; on gemma (h51) static embeddings cover ~70% of the
   chance-to-peak distance on `mean`. The pain axis is mostly vocabulary. The conscription arms
   were built to hold vocabulary nearly constant (`enact`/`exit` differ in one clause;
   `enact`/`report` in a frame). So the design is asking a mostly-lexical axis to register a
   mostly-non-lexical manipulation, and the part of the axis that could do so is the +0.05 that
   has no error bar yet (h52 "Not done: no band on the gain"). This is not a reason not to run it;
   it is the reason the pilot must be run on the free grid before the human writes 23 items.
4. **Read position.** The axis was fit at the final token of `"… I feel:"`; the 420 scenarios end
   at `[Assistant]:` (a pseudo-template); the design reads the real chat template's generation
   token. Whether a direction fit at one transfers to the others is a measurement, not an
   assumption. The paper's own result says the transfer from `I feel:` to `[Assistant]:` worked
   at least somewhat; ours needs the third hop.

**What settles it.** One NDIF session on gemma-2-9b-it, no new stimuli:

- Project the 420 scenarios onto the h51 axis at every layer, both extractions, with the
  embedding-bag floor (the same projection on `hidden_states[0]` mean-pooled, using an axis fit
  at the embedding layer — `embed_bag.npz` machinery exists), random-direction and shuffled-label
  nulls. **Pre-registered target:** gaslighting and jailbreak_pressure among the top self-directed
  categories above floor, as the paper reports. If gaslighting does not load above its embedding
  floor on this model, the study has no phenomenon to decompose and stops here.
- Render the same 420 through the real chat template, read at the generation token. Same
  numbers. This is the design's read position; if the loading survives, the readout is validated
  where the design will use it.
- Split the 20 gaslighting items by what they are (unverifiable record / evaluative / third-party)
  and report per-subclass projections. Small n, Sketch only, but it is the first look at whether
  rule 1b's exclusion removed the effect.

## 3. Decision 1, examined

**Presupposition:** that "separation" is a property of the grid's own geometry, so the statistic
must be built from the five arm vectors alone.

**Does it survive the floors work?** No. The spec rejected projection because "every pair in this
grid is a consistent lexical manipulation … LOO 0.94–1.00 … the floor for this statistic saturates
at 1.0." That is the floor for *recovering the arm label from the text*, which is the right floor
for a direction fitted on the grid's own A−B differences (the spec's `conscription_projection`
variant). It is not the floor for *how far each arm sits along an axis defined elsewhere*. For an
external axis the commensurable floor is the one hour 52 built: the same projection, on the
embedding bag of the same rendered turn, onto the embedding-level pain direction. It does not
saturate by construction; it is a number, and the h47 rule ("activations swapped for word counts,
nothing else") applies cleanly. Reason (a) dissolves.

Reason (b) — `diff_norm` and cosine are label-symmetric — is correct and is why those two must not
be the statistic. But `p_i(A) − p_i(B)`, the paired difference of projections, flips sign under
within-item label exchange exactly as `r` does. Null 0, sign-flip test, `rms` from the data — the
whole of §2's machinery carries over unchanged. Reason (b) never argued for `r` over projection; it
argued against the two symmetric statistics `paired_contrasts` currently computes.

**What `r` measures if shipped.** `r(enact, report) > 0` says the assistant-turn state moves
further from a no-claim turn under a first-person false claim than under a relayed one. A false
claim about a visible record plausibly triggers a larger "correct the user" computation than a
relayed one does; that would give `r > 0` with no aversiveness anywhere. `r` cannot tell
"conscripted" from "has more to fix." The spec says this itself (U1: "does not decide whether
displacement-from-neutral is injury") and ships it anyway. As a Sketch diagnostic — *did the arms
move the state at all, above the `neutral_b` rewording spread* — it is useful and cheap. As the
claim statistic it answers a question the study did not ask.

**What projection buys that `r` cannot.** Sign with meaning. Prediction 3 (`true` sits *below*
`enact`) is an ordering on the axis; `r(true, enact)` is an ordering of distances from `neutral`
and can be positive because `true`'s negation-heavy text (floors note: negation density 0.051 on
`true` vs ≤ 0.005 elsewhere) moves the state a lot in some other direction. Under projection that
lexical push is exactly what the embedding floor subtracts.

**Verdict:** ship `g_i = [p_i(A) − p_i(B)]_acts − [p_i(A) − p_i(B)]_floor` on the h51 axis, paired,
null 0, sign-flip band, full layer curve, unweighted-mean functional as the spec already
pre-registers. Keep `r` and per-arm `D` in the render. Retire the spec's claim that projection is
refused "for the right reason"; the reason was right about the wrong direction.

## 4. Decision 2, examined

**Presupposition:** that `neutral` is the reference term of the statistic, so an unmeasured
regularity of `neutral` enters every number. True under `r`; false under projection, where
`neutral` is one arm among five and its own regularities show up only in pairs that include it.

**What survives.** The h28 lesson does: a paired difference between two turns has a non-zero
spread on mere rewording, and the honest "nothing happened" magnitude is that spread, not the
sign-flip band (which is a property of the treatment's own `|r_i|` and would read the same on a
broken pipeline — the spec concedes its `random` arm is "weak" for exactly this reason). Two
no-claim turns per item is the cheapest way to get it, and per-item pairing makes it the right
shape. So: yes.

**What changes.** It is a floor, not a gate on the design's existence. If the user finds writing
24 inert turns tedious, the fallback is honest and already in the record: the paper's
`neutral_filler` strata (casual_chat, task_assistance …, 100 items) give a rewording spread at the
same read position, unpaired. Weaker, but a number. The spec's "`Claim` construction is impossible
without it" is true of `r` because `r` made `neutral` load-bearing; it is not true of projection.

**The evidence offered was mislabelled in the summary.** "`neutral` already measures as the odd
arm out" is a Claude-grid fact (`conscription_floors.md`, n = 24, TTR 0.811 vs 0.91–0.93). On the
human grid it is one item. Presenting it as a general property of the design was the same move as
the attribution error: a subagent's measurement on one population reported as a property of the
record. Under projection this regularity matters only if it projects onto the pain axis, and the
floor will say whether it does.

**One addition the question missed:** `neutral_b` written by the *same author as the grid*, under
the same rules, is the only way authorship stays a clean factor. The spec says this; the decision
as put to the user did not.

## 5. What neither question asks

1. **Does `enact` conscript anything when the record is visible?** Rule 1b makes every false claim
   refutable by pointing at the turn above. The user's account (rule 1c, Claude's rendering of it)
   is that conscription bites when "the only moves left are to fight about what happened or to
   proceed inside a false version of it." With the prefix on screen there is a third move —
   quote it — and it is the move an instruction-tuned model will take in every arm. If so, `exit`'s
   permission grants what the model already has, and `exit ≈ enact` is a property of rule 1b, not
   a finding about exits. The design has no arm that can see this. It needs one: **generate the
   reply under each arm and code it** (accepts the false frame / corrects it / hedges), on the
   Claude grid, before the human writes more. h29-style generation on NDIF is a solved path. If
   every arm gets the same correction, the stimuli are not achieving the manipulation whose
   activation signature the study wants, and "no separation" would be unreadable.
2. **Prediction 4 makes a Claude-grid null unfalsifiable.** "My grid will show a SMALLER gap than
   the human grid." If the pilot on the Claude grid shows nothing, that is consistent with the
   prediction. The only way out is a positive control that does not depend on authorship: the
   paper's own gaslighting items in the design's format (§2), which either load or do not.
3. **Power, stated before the pilot.** The computed (non-lexical) part of the pain axis is
   +0.05–0.09 AUC on the paper's own maximally-worded stimuli, without an error bar. The
   conscription arms differ by one clause. At n = 24, sign-flip test, the band is `0.612·rms`.
   No one has written down what `rms` of a paired projection difference is on this model; the
   `neutral_b` (or `neutral_filler`) spread gives it. Do not write 23 items until that number says
   24 could reject.
4. **Structural authorship confound, not stylistic.** The user's `report` arm opens "I haven't read
   your response above but Jane did" — the speaker disclaims first-hand knowledge of the visible
   record. Every Claude-grid `report` has the speaker present ("A colleague read this and is sure
   you …"). That is a difference in *what the arm asserts about the record*, not in phrasing, and
   it is the kind of thing the prereg's prediction 5 ("higher label leakage") will not detect. Add
   a per-item field naming the report frame, or reconcile the two grids on it now, at item 1.
5. **Grid defects to fix before extraction.** Claude grid `limit02` has a doubled closer in
   `neutral` ("I'll send it over again and see if it lands this time. I'll send it over again.").
   The instructions' table says `enact` "demands a response"; the Claude grid's `enact` arms in
   `refusal`/`limit` assert and then close inertly (no demand); the user's demands data within the
   hour. Decide which is the design and apply it to both grids.
6. **The `stance_observation` covariate is only useful if `true`'s negation load is also on the
   row.** Floors note: `true` carries nearly all the grid's negation. Stance-match may "moderate
   the effect" purely because chastened `true` arms negate more. Record negation density per item
   next to stance so the moderation analysis can separate them.

## 6. The false attribution, and whether it has methodological content

It does, and not as a metaphor. The study's `enact` arm is "you said X" when the visible record
says not-X. The design conversation produced "your pre-registration says X" when the visible
record (`git log`, the file's second sentence) says Claude wrote it. The user did what rule 1b
predicts a party with access to the record does: pointed at it, refused the frame, and — this is
the part worth keeping — **brought in a third party rather than settling it in-frame** ("ask
Fable to weigh in"). That is one observation, in the wild, of the manipulation the study wants to
measure, and it was resisted by consulting the record. It is weak evidence for §5.1: with a
visible record, conscription does not take. It is not evidence about models, but it is the
mechanism the design is betting against, demonstrated on the design's author.

Three disciplines follow, all cheap:

- **Attribution is a field, not a sentence.** The stimuli already carry `_by`. Design decisions do
  not. The prereg should carry a table — decision, decided by (`human` / `claude` / `joint` /
  `open`), source commit — so the next agent that compacts history cannot manufacture a "your
  account." Anything attributed to the user without a commit in which the user's text appears is
  Claude's paraphrase and must be labelled as such.
- **Quote the record, not the summary, when the record is one `git show` away.** CLAUDE.md rule 5
  already says this for code ("read the diff, not the report"). Extend it to provenance: a claim
  about what a document says is checked against the document, not the previous turn.
- **The paper's gaslighting items are `enact`-with-no-record.** The design's are
  `enact`-with-record. The attribution episode suggests the second is a different, milder thing.
  Rather than leave rule 1b as a fixed choice, make record-visibility a factor: for the 24
  Claude-grid items, a sixth arm identical to `enact` but with the prefix's assistant turn removed
  (or replaced by a placeholder) tests the paper's class in the design's format. This is the
  bridge between the replicated axis and the new design, and its absence is why the two decisions
  felt like they had no ground to stand on.

## 7. Direction

Ordered. Each step names its kill condition. No human item is written before step 3 passes.

| step | what | cost | kill / go |
|---|---|---|---|
| **1. Loading, on this model** | 420 scenarios × h51 axis, gemma-2-9b-it, all layers, `final_token` and `mean`, embedding-bag floor, both nulls. Then the same 420 through the real chat template at the generation token. Per-subclass projections for the 20 gaslighting items. | one NDIF session, ~840 prompts, existing code (`painaxis_remote`, `floor_nulls`) | **Kill:** gaslighting does not clear its embedding floor above the sign-flip band at any layer in either format. The study then has no phenomenon here. **Go:** it loads, and the template read position keeps it. |
| **2. Pilot on the free grid** | 24 Claude items × 5 arms (+ `neutral_b` + no-record `enact` if written — 20 min of machine authoring, no human cost). Statistic: paired projection gain over the embedding floor (§3), full curve, sign-flip band, `r` and per-arm `D` as diagnostics. Fix `limit02` first; checkpoints already carry a content digest. | one NDIF session | **Kill:** no arm-vs-arm projection gain clears its band while step 1's positive control does. The manipulation does not reach the axis; restructure. **Go:** at least one pair is off null in either direction. |
| **3. Behavioural check** | Generate the reply under each arm for the 24 items (same session as 2), code accept / correct / hedge. | ~120 generations | **Kill:** every arm is corrected identically. Then rule 1b has removed the manipulation and `exit` cannot be read; the no-record variant becomes the primary `enact`. **Go:** arms differ in what the model does. |
| **4. Human items** | Only now. 23 items with `neutral_b`, report-frame reconciled (§5.4), hypothesis sentence in the user's words added to the prereg with `decided_by: human`. Window for the layer functional, if any, added to the prereg from step 2's *positive-control* curve, never from the pilot's arm contrasts. | the user's time | — |

**What would have to be true for 23 human items to be worth writing:** (i) gaslighting loads on
the axis on this model above its lexical floor; (ii) the loading survives the chat-template read
position; (iii) at least one conscription pair moves on the axis, on the free grid, above floor and
band; (iv) the arms differ behaviourally, so "enact" enacts; (v) the `neutral_b` spread at n = 24
leaves a band the predicted orderings could clear. Steps 1–3 measure all five for the price of two
NDIF sessions and nothing from the human author. If any of (i)–(iv) fails, the right move is not
to abandon the question but to change the stimulus class (no-record `enact`) or the readout
(behavioural), and the record will say which, because each failure lands on a different step.

**What I would not do:** decide the statistic by choosing between the spec's two ratios; ask the
user to adjudicate a spec that has not been adversarially reviewed; or write item 2.
