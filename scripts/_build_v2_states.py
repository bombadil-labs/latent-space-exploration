"""Builds prompts/time_translation_v2.json from v1 by replacing the far-Dt
(>=100 years) state spans with vocabulary-disjoint rewrites, one per subject.
Near-Dt (t0..1year) states, phrases, controls are untouched (copied from v1).
Run: python scripts/_build_v2_states.py
"""
import json, copy

V1 = json.load(open("prompts/time_translation_v1.json"))
FAR = ["100years", "1000years", "10000years", "1000000years"]

NEW = {
"street": {
"100years": [
"A century later Alder Street still appears in the city's address register, though the frontages have changed hands many times. Zoning has shifted from mixed retail to residential-over-shopfront, and the plot boundaries drawn in the original survey remain the basis for every subsequent permit.",
"After a hundred years the address persists in the land register even where the building does not. Three redevelopment permits have been filed for the block since the original survey, each keeping the old plot lines and each renaming a shopfront.",
"A hundred years of planning applications have kept Alder Street's number in continuous use. The frontages have been rebuilt twice under permit, the shop mix has turned over completely, and the plot boundaries on today's title deeds trace directly back to the original survey.",
],
"1000years": [
"A thousand years on, the address survives only as an entry in an archived land register, if any register from that century still exists at all. The plot lines a modern surveyor would draw bear no necessary relation to the ones filed under the street's name.",
"After a millennium, whatever administrative system tracks addresses has itself been replaced many times over; the name Alder Street may persist as a heritage designation without any corresponding register entry. No permit history connects a current parcel to the original one.",
"Ten centuries of municipal reorganisation make the original address unrecoverable in any land register now in force. A plot might still be numbered along that alignment, by local custom, but no filed survey ties it to the street's original boundaries.",
],
"10000years": [
"Ten thousand years later no administrative system in continuous existence could reference the original address; every land register, permit archive and surveying convention has been superseded and superseded again. Whether a settlement occupies the site is a separate question from whether it is Alder Street.",
"After a hundred centuries the concept of a filed plot boundary presumes an institution with no plausible continuity across that span. The address is not lost so much as the entire practice of addressing has lapsed and been reinvented, repeatedly, by then.",
"Ten millennia outlast any single system of land registration. Whatever people occupy that ground, if any, will use their own convention for marking a parcel, unrelated to plots, permits or the numbering Alder Street once had.",
],
"1000000years": [
"A million years later the question of the address has no institutional answer: registers, permitting authorities and surveying conventions rise and fall on timescales of centuries, not that of geological time. Whether a system of marking a parcel exists on that ground is itself uncertain.",
"After a million years no register, permit or planning authority from any human era persists, several times over. A parcel of ground near the original coordinates might exist, but nothing about it would be filed, numbered or zoned in any sense continuous with Alder Street.",
"A million years is long enough that the very practice of registering an address may not persist among whatever occupies that ground. The street's number, its zoning and its permit history belong to an administrative order with no claim on that distant a future.",
],
},
"mountain": {
"100years": [
"A century later Caldreth's summit height, resurveyed by theodolite, matches the original trig-point figure to within a few centimetres. The granite of the upper cirque shows fresh frost-shatter, and a new benchmark has been set beside the old one.",
"After a hundred years a repeat survey finds the ridge line and summit elevation essentially unchanged from the first trig reading. Frost has plucked a few blocks from the cirque headwall, and the old benchmark has been re-levelled rather than replaced.",
"A century of resurveys puts Caldreth's height within the error bars of the original trig measurement. The granite ridge has shed loose blocks into the cirque, and a fresh benchmark sits a few metres from where the first one was driven.",
],
"1000years": [
"A thousand years on, successive resurveys show the summit a metre or so lower than the first trig figure, the cirque headwall stepped back by frost-plucking, and the granite ridge measurably blunter along its crest.",
"After a millennium, repeated triangulation shows Caldreth's elevation down by roughly a metre, the cirque enlarged, and a new moraine lobe built out below the old one where a glacial advance and retreat has come and gone.",
"Ten centuries of resurvey put the summit a metre lower and the ridge crest measurably rounder. The cirque has widened by frost action, and a moraine now sits where bare granite once was.",
],
"10000years": [
"Ten thousand years of resurvey show Caldreth's summit down several metres and the cirque enlarged into a broader bowl by more than one glacial advance. The ridge crest has been backworn, and successive moraine lobes record each ice episode in sequence.",
"After a hundred centuries, triangulation shows the peak several metres lower than its first recorded height, the granite ridge measurably thinner, and the cirque doubled in width by repeated glacial occupation and retreat.",
"Ten millennia of survey data show the summit down by several metres, the cirque reshaped by more than one glacial cycle, and a stack of moraine lobes at different distances marking each advance in turn.",
],
"1000000years": [
"A million years of denudation, at roughly a tenth of a millimetre a year, have taken on the order of a hundred metres off Caldreth's summit; the sharp granite ridge of the first survey has been reduced to a broad, rounded shoulder.",
"After a million years, cumulative denudation has lowered the peak by something like a hundred metres and merged the once-separate cirques into one wide bowl; no single trig position from the original survey would still mark the true summit.",
"A million years of frost, ice and running water have denuded Caldreth by on the order of a hundred metres. The granite core once capped by a sharp ridge now presents a broad shoulder, and the original trig position sits well below the current summit.",
],
},
"orchard": {
"100years": [
"A century later the original cultivar survives only where a grower deliberately propagated scions from it; commercial planting has moved through four other varieties in the interval. The six-hectare block itself has been replanted twice under different rootstock.",
"After a hundred years, nursery catalogues show the original variety maintained in one heritage collection, propagated by grafting rather than by seed. On the ground, the block has passed through arable use and two other rootstock generations.",
"A century of orchard turnover has retired the original cultivar from commercial planting; it persists, if at all, as grafted material in a pomological collection. The field itself now carries a different rootstock and variety mix entirely.",
],
"1000years": [
"A thousand years on, no nursery links any living tree to the original cultivar; grafting lineages this long are not maintained by any institution known to persist that far. The rootstock and scion combinations grown on that ground bear no traceable relation to the first planting.",
"After a millennium, the practice of grafting a named cultivar forward through a thousand years of nurseries has no precedent and no surviving chain of custody. Whatever fruit grows on that ground descends, at best, by open pollination many generations removed from the original variety.",
"Ten centuries outlast any pomological collection or grafting lineage. The named cultivar of the first planting has no institutional continuity into that period; seedling apples on or near the site would be unrelated hybrids, not propagated stock.",
],
"10000years": [
"Ten thousand years later the cultivated apple as a managed variety has almost certainly gone through many rounds of domestication, abandonment and re-domestication elsewhere; no lineage traceable to the original scion wood persists. The land use of that particular field is unknowable.",
"After a hundred centuries, agriculture itself as a continuous institution capable of maintaining grafted lineages cannot be assumed; the cultivar and the land-use designation of the original six hectares belong to a system with no claim on that timescale.",
"Ten millennia is longer than the entire history of orchard cultivation to date. Whether apples are grown at all on that ground, under any variety name, is a separate and unanswerable question from whether the original cultivar exists anywhere.",
],
"1000000years": [
"A million years later the cultivar, its rootstock lineage and the species of apple it belongs to have most likely all been superseded by evolutionary and agricultural change many times over; no scion wood or grafting chain spans that interval.",
"After a million years, domesticated agriculture as practiced when the orchard was planted is one of many agricultural systems that will have risen and lapsed; the specific cultivar and its genetic lineage are not expected to persist in any form.",
"A million years outlasts the species Malus domestica as presently constituted. Whatever fruit-bearing plants occupy comparable land will carry no cultivar name, rootstock lineage or grafting chain connected to the original planting.",
],
},
"mayfly": {
"100years": [
"A century later the species is still flying somewhere in the region, its numbers roughly halved by a century of nutrient enrichment and warming. A hundred generations separate the present population from the individual described; nothing about that one adult persists in any specimen collection.",
"After a hundred years, a hundred generations of drift and selection have passed for this species. Its distribution has shifted with regional water quality, and no genetic sample links any living individual to the one described; specimen labels of that decade, if any survive, note only the species, not the animal.",
"A hundred generations have followed the described mayfly. The species persists regionally at reduced density, its emergence timing shifted earlier by a fortnight of warming; the individual itself left no descendant that can be traced by any method.",
],
"1000years": [
"A thousand years later the species may or may not still occupy the region; a millennium of drift, selection and local extinction-and-recolonisation has passed. A thousand generations separate any surviving population from the described individual, which is a statistical non-entity at that remove.",
"After a millennium, whatever mayfly population exists regionally has been reshuffled by a thousand generations of genetic drift and range shift. No specimen, genetic or otherwise, connects it to the individual described; the lineage, if unbroken, has changed beyond recognition.",
"Ten centuries is a thousand generations for this insect. Its genetics, its emergence timing and very possibly its geographic range have all shifted measurably; the described individual's one day of adult life contributes nothing traceable to any population that far on.",
],
"10000years": [
"Ten thousand years later, ten thousand generations of drift, selection and range shift have passed for the species; it may have diverged into a distinct population, spread across a different range, or been locally extinguished and replaced by immigration. The individual described is meaningless at this remove.",
"After a hundred centuries the lineage has, in all likelihood, either speciated or been replaced regionally at least once. Ten thousand generations is ample time for measurable morphological and genetic divergence from the described population, if any direct descent persists at all.",
"Ten millennia is long enough for this species to have shifted its range across a continent and to have diverged genetically from any ancestor of the described individual by an amount a taxonomist would notice. The described mayfly's day of flight is an instant ten thousand generations back.",
],
"1000000years": [
"A million years later the order Ephemeroptera persists, as it has for some three hundred million years, but the described species almost certainly does not; a million generations is ample for speciation and extinction several times over. No line of descent can be traced to the individual described.",
"After a million years, mayflies of some kind still emerge somewhere, but taxonomically they are not the described species; a million generations of drift and selection dissolve any specific lineage. The individual's single day of adult life is an unmeasurable instant across that interval.",
"A million years outlasts the species entirely, though not the order. The described mayfly belonged to a population that has, over that timescale, either transformed into something a taxonomist would name differently or left no descendants at all.",
],
},
"asteroid": {
# already vocabulary-disjoint in v1 (orbital mechanics register); carried over verbatim
"100years": V1["states"]["asteroid"]["100years"],
"1000years": V1["states"]["asteroid"]["1000years"],
"10000years": V1["states"]["asteroid"]["10000years"],
"1000000years": V1["states"]["asteroid"]["1000000years"],
},
"river": {
"100years": [
"A century later gauge records show the channel migrated some tens of metres across its floodplain, with two meander bends cut off into oxbow ponds. Discharge and channel width are statistically the same as in the first survey.",
"After a hundred years of monitoring, the thalweg has shifted position within the same floodplain corridor; two bends have avulsed into cut-off ponds, but channel width, gradient and bed material match the original gauge data.",
"A century of channel migration has moved this reach across its own floodplain by bend cutoff and lateral shift, leaving crescent-shaped former channels in the pasture. Gauge readings of width and discharge are unchanged from the first survey.",
],
"1000years": [
"A thousand years later the channel occupies an entirely different position within the same floodplain corridor, having swept back and forth across it repeatedly; a metre or more of overbank silt has been laid across the location of the original reach.",
"After a millennium of avulsion and lateral migration, the present channel bears no positional relation to the surveyed reach; the floodplain records a braid of abandoned courses, and channel pattern, width and gradient are the only quantities that remain comparable to the original gauge data.",
"Ten centuries of channel migration make the described reach unlocatable except by excavation of the floodplain. The corridor, the discharge regime and the meander wavelength are statistically the same; the position of any given bend is not.",
],
"10000years": [
"Ten thousand years later the drainage's discharge regime and sediment load have shifted with regional climate; the channel has incised in one phase and aggraded in another, and its planform through the corridor bears no relation to the surveyed one. Terraces record the intervening phases.",
"After a hundred centuries the river's channel pattern may have switched between meandering and braided more than once as climate and discharge shifted; the surveyed reach corresponds to no locatable feature, though the corridor itself has not moved.",
"Ten millennia has taken the drainage through repeated shifts in discharge and channel pattern; terraces at several levels record former floodplains, and no gauge reading from that period would resemble the original survey except in the identity of the corridor.",
],
"1000000years": [
"A million years later the drainage network itself has likely been reorganised by tectonic uplift and stream capture; the corridor may carry a different river, flowing a different direction, or none at all. Kilometres of channel deposits mark where earlier rivers, including this one, once ran.",
"After a million years the described river exists only as a body of channel and floodplain deposits somewhere in the stratigraphic record; drainage reorganisation and tectonic uplift have very likely redirected water elsewhere across that corridor.",
"A million years outlasts the river as a continuous drainage feature; stream capture and uplift have probably rerouted the network more than once, and the surveyed channel corresponds to a stratigraphic unit rather than to any water currently flowing.",
],
},
"real_population": {
# 100 years unchanged (already institutional/historical, not erasure vocabulary)
"100years": V1["states"]["real_population"]["100years"],
"1000years": [
"A thousand years later no demographic projection is meaningful; the recorded trajectory to 1939 and the subsequent census figures already span less than a fifth of the interval. Whatever counting authority exists by then would use categories, boundaries and record-keeping conventions unrelated to the London County Council's.",
"After a millennium, extrapolating a population figure from the 1801 census requires assuming continuity of the enumeration itself for ten times longer than any such institution to date has lasted; no such continuity can be assumed.",
"Ten centuries on, no census bureau, borough boundary or vital-registration convention from 1801 persists in any form a demographer would recognise. A number could be guessed for that year, but it would rest on no institutional continuity with the original count.",
],
"10000years": [
"Ten thousand years later there is no basis for a population figure, because there is no basis for assuming any continuous counting institution at all; ten millennia exceeds the span of every census bureau, empire and written language that has existed so far by a wide margin.",
"After a hundred centuries, whoever occupies the Thames valley, if anyone does, will count people, if they count at all, using categories with no correspondence to 'London', 'England' or 'census'; the 1801 figure is not a starting point for any chain of institutional succession that long.",
"Ten millennia removes not just the figure but the entire apparatus of enumeration that produced it: no register, bureau or state boundary from 1801 has ever persisted a tenth that long, and there is no reason to expect one to start now.",
],
"1000000years": [
"A million years later a population figure presumes continuity of counting, language and political organisation across a span many times longer than the species Homo sapiens has existed; no such continuity can be assumed, so no number attaches to London for that year.",
"After a million years, the entire institutional chain of parish registers, censuses and nation-states that could connect any figure to the 1801 count has long since been superseded many times over; whether people capable of counting still occupy the Thames valley is itself an open question.",
"A million years exceeds the history of settled counting by orders of magnitude. No census, register or state from 1801 has any claim on that year, and the question of London's population becomes, at that remove, a question with no institutional referent.",
],
},
"fictional_population": {
"100years": V1["states"]["fictional_population"]["100years"],
"1000years": [
"A thousand years later no demographic projection is meaningful for Veyle either; the recorded trajectory to year 751 and the following figures span a small fraction of the interval. Whatever counting authority exists by year 1612 of the Ash Reckoning would use categories, wards and conventions unrelated to Veyle's founding enumerators.",
"After a millennium, extrapolating from the year-612 count requires assuming an unbroken chain of enumerators, wards and calendars for ten times longer than the Ash Reckoning itself has run; no such chain is plausible.",
"Ten centuries on, no census office, ward boundary or vital-registration convention from year 612 persists in any recognisable form. A figure could be guessed for that year, but nothing ties it institutionally to the original count.",
],
"10000years": [
"Ten thousand years later there is no basis for a population figure for Veyle either, because no counting institution, ward or calendar from year 612 has any plausible claim to persist that long; the Ash Reckoning itself is a small fraction of the interval.",
"After a hundred centuries, whoever occupies the Ossel delta, if anyone does, will enumerate people, if at all, by conventions unrelated to 'Veyle' or the Ash Reckoning; the year-612 count is not a link in any chain of institutional succession that extends that far.",
"Ten millennia removes the entire apparatus behind Veyle's figure: no ward, calendar or counting office from year 612 has ever run a tenth that long, and nothing suggests one would by then.",
],
"1000000years": [
"A million years later a population figure for Veyle presumes continuity of counting, calendar and political organisation across a span many times longer than the people who kept the Ash Reckoning have existed; no such continuity can be assumed.",
"After a million years, the entire chain of ward registers, census counts and dynasties that could connect any figure to the year-612 count has been superseded many times over; whether counting itself continues on the Ossel delta is an open question.",
"A million years exceeds the history of the Ash Reckoning by orders of magnitude. No register, ward or calendar from year 612 has any claim on that year; Veyle's population becomes a question with no institutional referent at that remove.",
],
},
}

# ---- assemble v2 grid: copy v1, splice in NEW states + regenerate "prompts" for far Dt ----
V2 = copy.deepcopy(V1)
V2["_note"] = ("v2 (spec docs/specs/time_translation_v2.md via hour-29 follow-up): identical to v1 "
               "except the far-Dt (>=100 years) state spans are rewritten per subject in that subject's "
               "own vocabulary (registers/permits for the street, survey/petrology for the mountain, "
               "cultivar/rootstock for the orchard, generations/lineage for the mayfly, orbital mechanics "
               "for the asteroid [unchanged from v1], channel/discharge for the river, census/enumeration "
               "for both populations) to remove the shared erasure/geological vocabulary "
               "(eroded, vanished, dust, sediment, gone, traces, forgotten, ruins, stone, ice) that v1's "
               "note flagged as a confound. control_prompts are unchanged (they use only the t0 state).")

for s, dts in NEW.items():
    for t, paras in dts.items():
        assert len(paras) == 3, (s, t, len(paras))
        V2["states"][s][t] = paras

FAR_SET = set(FAR)
for key, marked in list(V2["prompts"].items()):
    s, t, p = key.split("/")
    if t in FAR_SET:
        pi = int(p[1:])
        interval_phrase = V1["phrases"][t]
        state_text = V2["states"][s][t][pi]
        V2["prompts"][key] = f"[[interval: {interval_phrase}]] [[state: {state_text}]]"

# sanity: control_prompts untouched, t0/near-Dt prompts untouched
assert V2["control_prompts"] == V1["control_prompts"]
for key in V1["prompts"]:
    s, t, p = key.split("/")
    if t not in FAR_SET:
        assert V2["prompts"][key] == V1["prompts"][key]

json.dump(V2, open("prompts/time_translation_v2.json", "w"), indent=1)
print("wrote prompts/time_translation_v2.json")
