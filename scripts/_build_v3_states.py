"""Builds prompts/time_translation_v3.json from v2: t0 states are copied unchanged; every
Delta t >= 1 day state is rewritten so it never names, numbers or paraphrases its own interval
(no duration words, no "N years/days/..." construction, no token shared with the interval
phrase). Subjects, intervals, paraphrase count, phrases and control_prompts are unchanged.
Run: python scripts/_build_v3_states.py
"""
import json, copy

V2 = json.load(open("prompts/time_translation_v2.json"))
DT = V2["deltas"]

NEW = {
"street": {
"1day": [
"Alder Street looks exactly as it did: the baker has sold out by ten, a van has scraped the chemist's downpipe, and someone has chalked fresh prices on the pub board. Nothing here would show up in a photograph.",
"The street is unchanged in every particular that matters: the same shutters go up, the same vans double-park, the same four blocks of brick. Only the litter in the gutter is new, and a delivery crate left outside the cobbler's door.",
"The street is indistinguishable from before. Bread, shoes and medicines are sold in the same doorways, the awnings are a shade more faded, and a fresh scuff marks the kerb where a lorry mounted it this morning.",
],
"1week": [
"Alder Street carries small news but no structural change: the cobbler is shut with a bad back, scaffolding has gone up over the chemist, and the pub has a new sign. The trade, the traffic and the buildings are as they were.",
"Scaffolding now stands against a shopfront, a single business has stayed dark through the market's usual cycle, and the stalls have turned over their goods again. The brick, the awnings and the plane trees are exactly as before.",
"Same buildings, same tenants: a shutter is down for illness and another facade is under scaffolding for repointing. Rubbish has been collected on schedule. Nothing that would appear on a map has altered at all.",
],
"6months": [
"The season has turned: the plane trees are bare, awnings rolled in, braziers standing outside the pub. The chemist has been repainted and the cobbler's lease has passed to a phone-repair shop; everything else holds.",
"The season is inverted: cold light, bare branches, salt on the pavement. One shopfront has changed hands and another has been repainted, but the four blocks of brick, the kerb line and the trade of the street are what they were.",
"The street has crossed into the opposite season. Where there were leaves and shade there is now bare timber and early dark. Of the shops, one has closed and reopened under a new name; the rest keep their doorways and their trade.",
],
"1year": [
"Alder Street has come back round to its own season: leaves out, awnings down, vans in the lane at noon. Two of the twelve shopfronts have changed hands, and the chemist has extended into the unit beside it.",
"The street sits where it began, the same light on the same brick. Turnover has been modest: a phone-repair shop where the cobbler was, a cafe where the stationer was, and new paint on three facades. The street's shape is untouched.",
"The street is recognisably itself, its season come round again. Roughly a sixth of the tenancies are new, the plane trees are a touch taller, and a pothole at the bridge end has been patched twice.",
],
"10years": [
"Nearly every tenancy has turned over. The baker and the chemist are gone; there are two cafes, a nail bar, a betting shop and three empty units. The corner pub has been converted into flats and the plane trees now shade the whole kerb.",
"The buildings stand but the street is not the same business. Most signs are new, rents have doubled, one block has been demolished and rebuilt in glass, and the market end has been pedestrianised. Only the alignment and the brick are continuous.",
"The shoe repair, the stationer and the pub have all closed, a supermarket has taken three units at once, and the canal end has been redeveloped. The trees are mature, the kerbs re-laid, the street name unchanged.",
],
"100years": [
"Alder Street still appears in the city's address cadastre, though the frontages have changed hands many times. Zoning has turned over from mixed retail to residential-over-shopfront, and the plot boundaries drawn in the original plat remain the basis for every subsequent permit.",
"The address persists in the land cadastre even where the building does not. Several redevelopment permits have been filed for the block, each keeping the old plot markers from the original plat and each renaming a shopfront.",
"Ongoing planning applications have kept Alder Street's number in continuous use. The frontages have been rebuilt twice under permit, the shop mix has turned over completely, and the plot boundaries on today's title deeds trace directly back to the original plat.",
],
"1000years": [
"The address survives only as an entry in an archived land cadastre, if any register from that era still exists at all. The plot boundaries a modern surveyor would draw bear no necessary relation to the ones filed under the street's name.",
"Whatever administrative system tracks addresses has itself been replaced many times over; the name Alder Street may persist as a heritage designation without any corresponding cadastre entry. No permit history connects a current parcel to the original one.",
"Repeated municipal reorganisation makes the original address unrecoverable in any land cadastre now in force. A plot might still be numbered along that alignment, by local custom, but no filed survey ties it to the street's original boundaries.",
],
"10000years": [
"No administrative system in continuous existence could reference the original address; every land cadastre, permit archive and surveying convention has been superseded and superseded again. Whether a settlement occupies the site is a separate question from whether it is Alder Street.",
"The concept of a filed plot boundary presumes an institution with no plausible continuity across so wide a gulf. The address is not lost so much as the entire practice of addressing has lapsed and been reinvented, over and over.",
"No single institution of land registration has any claim on so wide a gulf. Whatever people occupy that ground, if any, will use their own convention for marking a parcel, unrelated to plots, permits or the numbering Alder Street once had.",
],
"1000000years": [
"The question of the address has no institutional answer: cadastres, permitting authorities and surveying conventions rise and fall on timescales far shorter than deep time. Whether any system of marking a parcel exists on that ground at all is unknown.",
"No cadastre, permit or planning authority from any human era persists, several times over. A parcel of ground near the original coordinates might exist, but nothing about it would be filed, numbered or zoned in any sense continuous with Alder Street.",
"Whatever occupies that ground, the practice of registering an address may not persist among it at all. The street's designation, its zoning and its permit history belong to an administrative order with no claim on so remote a horizon.",
],
},
"mountain": {
"1day": [
"The mountain is identical. A few cubic metres of snow have slid from a gully, a block has come off the ridge and lodged in the talus, and the tarn stands a touch higher. No measurement at human scale would show a change.",
"Nothing has happened to the mountain that anyone could see. Clouds have crossed it, the snowline has moved a little with the sun, and a little rock has fallen. Its height, its faces and its outline are exactly as before.",
"Caldreth is untouched. The weather has changed, the shadows have swung round, and frost has worked a little further into a crack. The granite, the cirques and the summit ridge are unaltered.",
],
"1week": [
"The mountain shows only weather. Fresh snow lies above the treeline and has already begun to settle, and a small slab avalanche has run in the north cirque. The rock itself is unchanged.",
"Caldreth is the same mountain. Snow has fallen and partly melted, the tarn has risen and fallen, and a walker has rebuilt the cairn on the shoulder. Nothing structural has moved at all.",
"The differences are surface ones: fresh snow in the gullies, a swollen meltwater stream, fresh rockfall dust on the talus. Caldreth's height and shape are identical to a fine tolerance. The cirques hold the same ice they held before.",
],
"6months": [
"The season has turned on the mountain. The snow cap now reaches down to the treeline, the tarn is frozen, and the footpath is impassable. The rock beneath is unchanged; only its covering has inverted.",
"Caldreth is in its other condition: white to the shoulder, the cirques loaded, the streams silent under ice. Freeze and thaw have prised a little more scree onto the apron. The form of the mountain is the same.",
"The mountain wears winter instead of summer. Snow depth on the north face has grown considerably and the summer path is gone. Measured as rock, Caldreth has lost only a negligible fraction of its mass to frost shatter.",
],
"1year": [
"The mountain is back where it started: snow cap reaching into the warm season, tarn open, path clear to the shoulder. A modest tonnage of rock has moved from face to talus, which is nothing against its bulk. Its height is unchanged.",
"Caldreth is indistinguishable from its first description. The season has completed a circuit and returned. Erosion has lowered the summit by a fraction of a millimetre and widened a crack; no survey at this scale could detect it.",
"The mountain is exactly as it began. The same gullies hold the same late snow and the same streams leave the talus. Only the tally of fallen blocks at the foot has grown very slightly.",
],
"10years": [
"There is no visible difference to the mountain. The glacierette in the north cirque has retreated some tens of metres, a rockfall has scarred one buttress, and the treeline has crept a little upslope. The summit is the same height.",
"Caldreth is unchanged in form. Its ice is smaller, its talus slightly deeper, and a marked path has been worn into a gully by boots. Photographs of it would need annotation to tell apart.",
"Weather has removed a few millimetres from the summit and a modest tonnage from the faces. The mountain is the same mountain: the same cirques, the same ridge, the same profile against the sky.",
],
"100years": [
"Caldreth's summit height, resurveyed by theodolite, matches the original trig-point figure to within a few centimetres. The granite of the upper cirque shows fresh frost-shatter, and a new benchmark has been set beside the old one.",
"A repeat survey finds the ridge line and summit elevation essentially unchanged from the first trig reading. Frost has plucked a few blocks from the cirque headwall, and the old benchmark has been re-levelled rather than replaced.",
"Resurveys put Caldreth's height within the error bars of the original trig measurement. The granite ridge has shed loose blocks into the cirque, and a fresh benchmark sits a few metres from where the first one was driven.",
],
"1000years": [
"Successive resurveys show the summit a metre or so lower than the first trig figure, the cirque headwall stepped back by frost-plucking, and the granite ridge measurably blunter at its crest.",
"Fresh triangulation shows Caldreth's elevation down by roughly a metre, the cirque enlarged, and a new moraine lobe built out below the old one where a glacial advance and retreat has come and gone.",
"Resurvey puts the summit a metre lower and the ridge crest measurably rounder. The cirque has widened by frost action, and a moraine now sits where bare granite once was.",
],
"10000years": [
"Resurvey shows Caldreth's summit down several metres and the cirque enlarged into a broader bowl by more than one glacial advance. The ridge crest has been backworn, and successive moraine lobes record each ice episode in sequence.",
"Triangulation shows the peak several metres lower than its first recorded height, the granite ridge measurably thinner, and the cirque doubled in width by repeated glacial occupation and retreat.",
"Survey data show the summit down by several metres, the cirque reshaped by more than one glacial cycle, and a stack of moraine lobes at different distances marking each advance in turn.",
],
"1000000years": [
"Denudation, at a slow steady rate, has taken on the order of a hundred metres off Caldreth's summit; the sharp granite ridge of the first survey has been reduced to a broad, rounded shoulder.",
"Cumulative denudation has lowered the peak by something like a hundred metres and merged the once-separate cirques into one wide bowl; no single trig position from the original survey would still mark the true summit.",
"Frost, meltwater and running water have denuded Caldreth by on the order of a hundred metres. The granite core once capped by a sharp ridge now presents a broad shoulder, and the original trig position sits well below the current summit.",
],
},
"orchard": {
"1day": [
"The orchard is in the same part of its cycle. A little more blossom has opened, a little has fallen, and the bees have kept working. Nothing about the trees or the rows has changed.",
"The blossom is perhaps a step further along: more petals on the ground, the earliest flowers already setting. The grass is a touch taller. Otherwise the six hectares are exactly as described.",
"The orchard shows only the ordinary progress of flowering. Pollination has continued, a row has been sprayed, and a cold night has damaged a few blossoms on the low ground. The structure is untouched.",
],
"1week": [
"The orchard has moved out of full bloom. Most petals are down, the alleys are white with them, and small green fruitlets are visible at the centres of the spent flowers. The trees and rows are otherwise unchanged.",
"The blossom is over and fruit set has begun. The hives have been moved to the next block, the grass has been cut again, and the first thinning is being planned. The layout of the orchard is identical.",
"The orchard has moved further along its cycle: petal fall complete, fruitlets forming, leaves now the dominant colour. Three hundred trees in twenty rows over six hectares, exactly as before. The hail nets remain furled.",
],
"6months": [
"The orchard is in the opposite part of its cycle. The fruit has been picked, the leaves are turning and falling, the grass has been left long, and the hail nets are furled. The rows stand bare and wet.",
"The orchard is post-harvest and shutting down. Bins are stacked at the headland, windfalls rot in the alleys, and the trees are dropping their leaves. Nothing is in flower and nothing is growing.",
"The orchard has inverted: blossom replaced by bare branches, mown green by leaf litter and mud. The crop is in store, pruning has not yet begun, and the irrigation has been drained down.",
],
"1year": [
"The orchard is in blossom again. The trees are a touch older, two have been replaced after canker, and the rest are marginally larger. Bees are working the rows and the alleys are mown as before.",
"The orchard has returned to the state it began in: white bloom, warm weather, hives at the row ends. Pruning has opened the canopies slightly and the last crop is long sold. It is recognisably the same picture.",
"The orchard has completed a circuit. It stands in flower again, on the same six hectares in the same twenty rows, with a few replacement whips where old trees failed. Only the rings in the wood record the passage.",
],
"10years": [
"The orchard is past its best. The trees are older, tall and biennial in their bearing, and yields are down by a third. Four rows have been grubbed out and replanted with a newer variety on dwarfing stock.",
"The planting has aged. Canker and woolly aphid have taken a dozen trees, the canopies have closed over the alleys, and the grower has begun a phased replacement. The blossom still comes each spring, thinner than before.",
"A mature orchard has become an old one. Trunks are thick and mossy, the hail-net frames have rusted, and one block has already been replaced. The six hectares are still an orchard, still cropping, but not the same trees.",
],
"100years": [
"The original cultivar survives only where a grower deliberately propagated scions from it; commercial planting has moved through several other varieties. The six-hectare block itself has been replanted twice under different rootstock.",
"Nursery catalogues show the original variety maintained in one heritage collection, propagated by grafting rather than by seed. On the ground, the block has passed through arable use and other rootstock cycles.",
"Orchard turnover has retired the original cultivar from commercial planting; it persists, if at all, as grafted material in a pomological collection. The field itself carries a different rootstock and variety mix entirely.",
],
"1000years": [
"No nursery links any living tree to the original cultivar; grafting lineages this long are not maintained by any institution known to persist that far. The rootstock and scion combinations grown on that ground bear no traceable relation to the first planting.",
"The practice of grafting a named cultivar forward through so many nurseries has no precedent and no surviving chain of custody. Whatever fruit grows on that ground descends, at best, by open pollination many steps removed from the original variety.",
"No pomological collection or grafting lineage endures across so many cycles of replanting. The named cultivar of the first planting has no institutional continuity into this period; seedling apples on or close to the site would be unrelated hybrids, not propagated stock.",
],
"10000years": [
"The cultivated apple as a managed variety has almost certainly gone through many rounds of domestication, abandonment and re-domestication elsewhere; no lineage traceable to the original scion wood persists. The land use of that particular field is unknowable.",
"Agriculture itself as a continuous practice capable of maintaining grafted lineages is not a safe bet; the cultivar and the land-use designation of the original six hectares belong to a regime with no claim on so vast a timescale.",
"This span outlasts the entire history of orchard cultivation to date. Whether apples are grown at all on that ground, under any variety name, is a separate and unanswerable question from whether the original cultivar exists anywhere.",
],
"1000000years": [
"The cultivar, its rootstock lineage and the fruit lineage it belongs to have most likely all been superseded by evolutionary and agricultural change many times over; no scion wood or grafting chain spans so vast a gap.",
"Domesticated agriculture as practiced when the orchard was planted is one of many agricultural systems that will have risen and lapsed; the specific cultivar and its genetic lineage are not expected to persist in any form.",
"This stretch outlasts Malus domestica as presently constituted. Whatever fruit-bearing plants occupy comparable land will carry no cultivar name, rootstock lineage or grafting chain connected to the original planting.",
],
},
"mayfly": {
"1day": [
"The mayfly is dead. It mated in the evening swarm, the female dropped her eggs onto the riffle, and both spent adults fell onto the water within moments of each other. The body is drifting downstream in the surface film.",
"The animal's entire adult life is over. It moulted, flew, mated and fell; what remains is a spent husk on the water surface, being taken by trout and wagtails. Its eggs are on the gravel of the riffle.",
"The mayfly has been used up completely. The swarm rose, paired and collapsed, and the individual is now a dead insect in the drift with its wings flat on the water. Nothing of it is alive except its eggs.",
],
"1week": [
"There is no mayfly at all. The body was eaten or broke up in the current soon after death. On the riverbed its eggs have not yet hatched; the individual is simply absent from the reach.",
"The animal has ceased to exist as a body. Its remains entered the food chain quickly. The river still has its evening swarms, made of other insects that emerged after it; this individual is gone entirely.",
"Nothing physical of the mayfly persists. Scavengers and flow have dispersed it. What continues is the population: eggs on the gravel, nymphs of earlier cohorts in the silt, and a swarm on any warm evening.",
],
"6months": [
"The mayfly is long gone and its offspring are half-grown nymphs burrowing in the silt of the same reach. It is winter and there are no adults flying. The individual has been absent for nearly all of the interval.",
"Nothing of the individual remains in any form. Its eggs hatched and the nymphs are now overwintering in the riverbed. The adult itself was gone within moments, relatively speaking, of the first description.",
"The question concerns the river, not the insect. The mayfly's atoms are distributed through fish, birds and sediment, and its descendants are young nymphs in the mud. There has been no mayfly for a long stretch now.",
],
"1year": [
"A new cohort of the same species is emerging on the same riffle, but the original animal has long been gone. Its offspring are approaching their own emergence. Nothing of the individual is identifiable.",
"The evening swarm above the river looks exactly as it did and contains none of the same insects. The described mayfly ended in a single night; the pattern it belonged to has completed a full circuit without it.",
"The cycle has closed without the individual. The species is present, the riffle is the same gravel, the swarm rises on the same schedule. The mayfly described has been dead for almost the entire interval.",
],
"10years": [
"The mayfly is many cohorts gone. The population has persisted through droughts and a pollution incident that thinned it badly for one season. Nothing traceable to the individual survives except an unmeasurable share of ancestry.",
"The river still supports the species at reduced density; the riffle has shifted a few metres and the willow has fallen in. The described animal is one of many millions of such insects, now all of them dead.",
"The lineage continues; the individual has been absent for nearly the whole interval and has left no recoverable trace whatever, its whole adult existence having been a single brief span.",
],
"100years": [
"The species is still flying somewhere in the region, its numbers roughly halved by prolonged nutrient enrichment and warming. Many broods separate the present colony from the individual described; nothing about that one adult persists in any specimen collection.",
"Many broods of drift and selection have passed for this species. Its distribution has shifted with regional water quality, and no genetic sample links any living individual to the one described; old specimen labels, if any survive, note only the species, not the animal.",
"Countless broods have followed the described mayfly. The species persists regionally at reduced density, its emergence timing shifted earlier by warming; the individual itself left no descendant that can be traced by any method.",
],
"1000years": [
"The species may or may not still occupy the region; prolonged drift, selection and local extinction-and-recolonisation has passed. Many broods separate any surviving colony from the described individual, which is a statistical non-entity at that remove.",
"Whatever mayfly colony exists regionally has been reshuffled by countless broods of genetic drift and range shift. No specimen, genetic or otherwise, connects it to the individual described; the lineage, if unbroken, is scarcely recognisable as such.",
"So many broods separate this insect from the described one that its genetics, its emergence timing and very possibly its geographic range have all shifted measurably; the described individual's brief adult life contributes nothing traceable to whatever flies there today.",
],
"10000years": [
"Countless broods of drift, selection and range shift have passed for the species; it may have diverged into a distinct cohort, spread across a different range, or been locally extinguished and replaced by immigration. The individual described is meaningless at this depth of time.",
"The lineage has, in all likelihood, either speciated or been replaced regionally at least once. So many broods is ample time for measurable morphological and genetic divergence from the described cohort, if any direct descent persists at all.",
"This span is long enough for this species to have shifted its range across a continent and to have diverged genetically from any ancestor of the described individual by an amount a taxonomist would notice. The described mayfly's flight is an instant, countless broods back.",
],
"1000000years": [
"The order Ephemeroptera persists, as it has for an immense span, but the described species almost certainly does not; that many broods is ample for speciation and extinction several times over. No line of descent can be traced to the individual described.",
"Mayflies of some kind still emerge somewhere, but taxonomically they are not the described species; that many broods of drift and selection dissolve any specific lineage. The individual's single brief adult life is an unmeasurable instant across so vast a gulf of time.",
"This span outlasts the species entirely, though not the order. The described mayfly belonged to a cohort that has, over that timescale, either transformed into something a taxonomist would name differently or left no descendants at all.",
],
},
"asteroid": {
"1day": [
"The asteroid is unchanged except in position: it has advanced along its path around the sun and completed a couple of full turns on its axis. No feature of its surface has altered in any detectable way.",
"The rock is identical. It has advanced a small fraction of its path and completed a couple of rotations. Micrometeorites have struck it, adding nothing measurable to a surface already saturated with craters.",
"Nothing about the body has changed but where it is along its path. Its shape, mass, rotation rate and surface are the same to any precision we can state.",
],
"1week": [
"The asteroid has swept further along its path and rotated many times over. It is otherwise exactly as described: the same shape, the same craters, the same rubble. Space weathering over so short a stretch is immeasurable.",
"Only the orbital phase has moved, by a small fraction of the whole circuit. Its surface, its spin and its mass are unaltered by anything that has happened. There is nothing to run on a body with no active processes.",
"The asteroid is in a different place and no other respect. Solar wind has darkened its surface by an amount no instrument could resolve; the craters, ridges and boulders are where they were.",
],
"6months": [
"The asteroid has covered a modest arc of its path and rotated many times. Physically it is unchanged: the same size, the same rubble-pile structure, the same cratered face.",
"The only difference is position and the illumination and temperature that follow from it. There has been no geological activity of any kind. The object is the same object, further round its ellipse.",
"The asteroid has moved a fair distance along its path. Nothing has hit it larger than a grain. Its shape, spin period and surface remain as described, to the limits of measurement.",
],
"1year": [
"The asteroid has completed a modest arc of its path and is unchanged in every other respect. It has rotated many times over. No crater on it is new at any size a telescope could see.",
"The body has moved a further stretch round its ellipse and is otherwise identical. Unlike anything on a planet it has no seasons, no weather and no erosion; only its position has advanced.",
"The asteroid is exactly the rock it was, in a different part of the same ellipse. Its mass is the same to within a trivial margin of accreted dust and ejected material.",
],
"10years": [
"The asteroid has swung round its ellipse several times and is physically the same. Its spin has been altered by sunlight by a tiny margin, and a metre-scale crater may be new. Nothing else has changed.",
"The object is unchanged except in position and, immeasurably, in spin. Sunlight has been nudging its rotation the whole while, which at this size amounts to nothing. The surface looks identical.",
"The asteroid has moved several times round its ellipse and is otherwise untouched. No impact large enough to reshape it has occurred, and the regolith has been stirred by micrometeorites to a depth of microns.",
],
"100years": [
"The asteroid has swept round its ellipse many times and remains the same body. Its rotation period has drifted by an imperceptible amount under radiation torques, and its surface has been darkened very slightly by the solar wind.",
"The rock is unchanged at any scale that matters. Position and orbital phase are the only large differences; a few small craters have been added and about the same number erased by seismic shaking.",
"Nothing visible has happened in the belt. The asteroid is roughly a kilometre across, cratered and inert, on the same path with slightly precessed elements. The body's shape and mass are as they were in every regard.",
],
"1000years": [
"The asteroid has made a great many circuits and is still the same object. Its path has precessed measurably and its spin has been changed by a few percent. The body itself is unaltered.",
"The asteroid remains physically as described, with a slightly different spin state and a thoroughly shifted position along its ellipse. A few ten-metre-scale impacts may have occurred somewhere on it; the shape is the same.",
"Existence in the belt has left the rock intact. It has been struck by countless grains and nothing large. Its path has drifted within the resonance structure of the belt, and its surface is as it was.",
],
"10000years": [
"The asteroid has swept round its ellipse an enormous number of times. Its spin may have been substantially altered by radiation torques, and one crater of a few tens of metres is likely new. It is still the same body.",
"The object is recognisably identical. Its orbital elements have wandered, its rotation has spun up or down, and a handful of minor impacts have redistributed regolith. Nothing has broken it apart.",
"Nothing has produced a change that would alter the description. The asteroid is wherever its path has taken it, cratered as before, with perhaps one fresh bright scar and a slightly different rotation axis.",
],
"1000000years": [
"The asteroid has completed an immense count of circuits. It has probably suffered one impact large enough to leave a hundred-metre crater and reset its spin, and may have been rubbled further. It is still, most likely, there.",
"The body's history is dominated by chance. Collisional models give a body this size a lifetime far longer than the timescale considered, so it very probably survives, reshaped at the hundred-metre scale, its path drifted by the Yarkovsky effect.",
"Existence in the belt has moved the asteroid far from the described point on its ellipse and left the rock essentially intact: the same mass, the same composition, a somewhat rearranged surface and a different spin axis.",
],
},
"river": {
"1day": [
"The river is the same river. The gauge has risen slightly with overnight rain, some sand has moved along the bed, and a willow branch has lodged on a riffle. The channel is unchanged.",
"Nothing about the reach has altered that a map would record. Flow and turbidity are slightly different, some grains have shifted downstream, and a bank has lost a handful of soil. The meanders are where they were.",
"The Dorn has carried its ordinary load of water and sediment past this point. Its planform, width, depth and banks are identical. Only the level in the gauge and the position of the sand have changed.",
],
"1week": [
"The river has been through a small flood. Fresh silt lies on the lower floodplain, a bar has been reworked, and a slab of the outer bank has collapsed into the pool. The line of the channel is unchanged.",
"The reach shows the marks of a single high flow: scoured riffles, new wood caught in the bends, a re-graded point bar. The meanders themselves have not moved perceptibly at all. The banks, the pools and the willows stand exactly where they did.",
"The Dorn is the same forty-metre channel between the same banks. Some cubic metres of gravel have travelled a short distance and the water is clearer again. Nothing structural has changed. The reach would be mapped today with the same line as before.",
],
"6months": [
"The river is in its opposite regime: low summer flow, weed growing over the riffles, bars exposed and drying. The channel position is essentially unchanged and the bends have migrated by a few centimetres at most.",
"The reach has passed from flood season to drought season. Water covers half the bed width, the willows are in full leaf, and the pools are stagnant and warm. The planform is the same as before.",
"The same river looks different and is not: low and weedy instead of high and turbid, with cattle standing in the shallows. Bank erosion over the interval amounts to a few centimetres on the outer bends.",
],
"1year": [
"The Dorn has completed a full cycle and returned to the described condition. The outer banks have retreated by perhaps ten centimetres, a willow has fallen in, and a riffle has shifted downstream a metre or two.",
"The river is where it began in season and very nearly where it began in space. A single flood has trimmed the bends and added to the point bars. The map would not need redrawing.",
"The reach is recognisably identical: the same width, the same bends, the same pools. Measured carefully, each meander has crept outward by a small margin.",
],
"10years": [
"The meanders have visibly moved. The bends have migrated a metre or two outward, a point bar has become vegetated land, and a large willow has fallen and diverted flow into a new chute. The overall pattern holds.",
"The channel has shifted within its floodplain by a couple of metres at the most active bends, and one tight loop is close to cutting through. Width, depth and character are otherwise as described.",
"Repeated floods have reworked the details. Several bends have eaten into the pasture, one has been armoured with stone by the landowner, and the bed has coarsened in one section. The river is still the same river.",
],
"100years": [
"The meanders have migrated tens of metres and two loops have cut off, leaving oxbow ponds in the pasture. The reach is in a measurably different place on its floodplain, though it has the same width, gradient and style.",
"The Dorn has rewritten its course within the valley. Where the described bend was there is now a crescent of marsh, and the channel runs some way to the north. The river is unchanged in kind.",
"Repeated bend migration and two cut-offs have moved this reach across its own floodplain. Old channels show as damp lines in the grass. The river remains a forty-metre gravel-bedded meandering stream, in new positions on the same land.",
],
"1000years": [
"The river occupies the same valley and none of the same channel. It has swept back and forth across the floodplain many times, leaving a braid of abandoned loops, and has laid a thick blanket of silt over the old bed.",
"The Dorn's planform is largely new while its character persists. The floodplain is a palimpsest of former courses, and the present channel crosses the described location at right angles, or misses it altogether.",
"Extensive meandering has made the described reach unlocatable except by excavation. The valley, the gradient and the sediment supply are the same, so the river still meanders at the same scale, in different places.",
],
"10000years": [
"The drainage system has been reorganised. Climate shifts have altered its flow and sediment load; it has incised in one phase and filled in another. Its course through the valley bears no relation to the described one.",
"The Dorn is a different river in the same valley. Terraces record earlier floodplains, and the present channel may be braided rather than meandering, depending on the regime. The described reach is buried or eroded.",
"The river has passed through repeated glacial and interglacial behaviour. Its width, its pattern and even its outlet have changed more than once. Nothing of the described bends, pools or banks survives anywhere. The valley is the only thing that has stayed put.",
],
"1000000years": [
"The drainage has been rearranged by uplift and capture. The valley may carry no river at all now, or may carry one flowing the other way. A great thickness of debris has passed through; the described river is a stratigraphic unit.",
"The Dorn exists only as cross-bedded sands somewhere downstream. Tectonics and repeated glaciation have redirected the drainage, and a river of some kind runs nearby, unrelated in direction to this one.",
"The river has been outlasted as an entity. Its catchment has been captured in part by a neighbouring system and its valley overdeepened and partly filled. Nothing of the described channel is meaningful at this scale.",
],
},
"real_population": {
"1day": [
"The population of about a million has changed by a small margin: a modest number of births, a modest number of deaths, and a small net inflow of migrants. Nothing in the city's fabric or economy has altered.",
"London is the same city of a million. The arithmetic of births, deaths and arrivals moves the total by a negligible margin. The docks work, the markets open, the parish registers add a page.",
"The count is indistinguishable from a million. Coaches have brought in a few dozen people from the counties, ships have landed some more, and the burial grounds have taken their share of the dead.",
],
"1week": [
"The population has risen by a small margin, well within the noise of any contemporary count. The city is unchanged: the same parishes, the same trades, the same crowding. A cold snap has raised mortality slightly.",
"London is still a city of about a million. A modest number have been born, a modest number buried, and several hundred have arrived from the countryside and from Ireland. The city's built area and its trades are exactly as they were.",
"The total has moved by a negligible fraction. No institution, street or industry of the city has changed. The Bills of Mortality record the ordinary numbers and nothing else.",
],
"6months": [
"London has perhaps fifteen thousand more people than it did, a small percentage. The season has turned from winter to summer, mortality has fallen, and the building trades are working again on the northern fields.",
"The population is around a million and fifteen thousand. Growth has been steady and unremarkable. New terraces are going up in Marylebone and Southwark, and the river is thick with coastal coal traffic.",
"The city is measurably but not visibly larger. Some thousands have arrived and some thousands have died, netting a small percentage. Nothing about the shape or the business of London has changed.",
],
"1year": [
"London's population is a little above a million, and the first national census will shortly count it for the first time at a figure somewhat higher still, once the outparishes are included. Growth runs at a modest percentage, almost all of it migration.",
"The city has added some tens of thousands of people. The first national census is taken around now and gives London a figure above a million, confirming what everyone assumed: it is the largest city on earth.",
"The population has grown by a modest percentage. The built-up edge has crept a short distance north and east. The trades, the river and the institutions are what they were.",
],
"10years": [
"London holds roughly 1.3 million people. The enclosed docks have been built at the Isle of Dogs and Wapping, new squares and terraces have covered fields to the north, and a war with France has concentrated shipping and finance in the city.",
"The population is about 1.3 million, up by a third. Enclosed dock systems have opened, gas lighting is beginning, and the northern suburbs are under construction. The city has grown outward as well as denser.",
"London has perhaps 1.3 million inhabitants. Substantial growth shows as new streets, new docks and a visibly larger built-up area, though the core trades and the crowding of the old parishes are unchanged.",
],
"100years": [
"London holds about 6.5 million people in the county and the surrounding built-up area, and is still the largest city in the world. Railways, underground lines, sewers and suburbs have transformed it; the population has multiplied more than sixfold.",
"The city of a million has become a metropolis of six and a half million. Grand sewer works, an underground railway, a great riverside embankment and mile upon mile of terraced suburb belong to this era of industrial growth.",
"Industrial expansion has taken London from one million to roughly six and a half million. The built-up area extends far beyond the old parishes, tied together by rail; it is the capital of an empire and the largest city on earth.",
],
"1000years": [
"Any figure is speculation. London's recorded trajectory ran on to a peak near eight and a half million before a long decline and a subsequent recovery. Beyond that lies pure extrapolation.",
"The city's recorded history is a small part of what could still unfold: growth to nearly nine million, a plateau and decline, then renewed growth. What stands on the Thames further on cannot be stated.",
"The question passes out of demography entirely. London has already been occupied for a very long stretch; further growth, decline, flooding or abandonment are all live possibilities. The estuary is subsiding and the sea is rising, which sets the terms.",
],
"10000years": [
"There is no basis for a population figure. The Thames estuary as it exists at present is a recent product of its own history and will not persist indefinitely; sea level and climate will have moved far outside the historical range.",
"The continuity of the city cannot be assumed. The interval considered is as long as the whole of settled agriculture. Whoever lives on that part of the river will be as distant from Londoners as Londoners are from the first farmers.",
"The subject itself is removed. The population of London at that remove is not a small or a large number; it is a category error: neither the city nor the language nor the coastline can be projected that far.",
],
"1000000years": [
"The species that built the city has an uncertain future, and the site lies under a different sea or a different ice sheet. No population figure attaches to London because there is no London and, in all likelihood, no continuity with the people who made it.",
"The question is geological. Southern Britain will have been glaciated repeatedly; the Thames has changed its course under ice before and will again. Whatever occupies the region will be unrelated to the population described.",
"The interval is longer than the species Homo sapiens has existed. The city, its population and its records are a thin layer in the sediment of one interglacial. No number can be given, and the place itself will not be recognisable.",
],
},
"fictional_population": {
"1day": [
"The population of Veyle, some nine hundred thousand, has changed by a small margin: a modest number of births, a modest number of deaths, and a trickle of arrivals down the terrace roads. The city's fabric and trade are exactly as before.",
"Veyle is the same city of nine hundred thousand. The arithmetic moves the total by a negligible margin. The salt barges unload, the looming-halls run, the ward registers add a line.",
"The count is indistinguishable from nine hundred thousand. A few dozen have come down the terrace roads, a ship has landed more, and the ash-yards have taken their share of the dead. Nothing in the wards or on the quays is different this morning.",
],
"1week": [
"The population has risen by a small margin, far inside the error of any count the city can make. Nothing has changed: the same wards, the same trades, the same crowding. A fever in the low wards has lifted mortality slightly.",
"Veyle is still a city of about nine hundred thousand. A modest number have been born, a modest number buried, and several hundred have walked in from the terraces. The city's built quarter and its guilds are exactly as they were.",
"The total has moved by a negligible fraction. No hall, quay or ward of the city is different. The weekly ash-lists record the ordinary numbers and nothing more.",
],
"6months": [
"Veyle has perhaps thirteen thousand more people, a small percentage. The season has turned, the salt pans are working again, mortality has dropped, and builders are back on the north fields.",
"The population is near nine hundred and thirteen thousand. Growth has been steady and unremarkable. New tenement rows are going up beyond the Ossel gate and the harbour is thick with coastal traffic.",
"The city is measurably but not visibly larger. Some thousands have arrived and some thousands have died, netting a small percentage. The shape of Veyle is unaltered. The wards, the quays and the looming-halls are as they were.",
],
"1year": [
"Veyle's population is a little above nine hundred thousand, and the first full ward enumeration will shortly count it at rather more once the outer wards are included. Growth runs at a modest percentage, almost all migration.",
"The city has added tens of thousands of people. The Crown's first full enumeration is taken around now and confirms what was assumed: Veyle is the largest city on the coast by a wide margin.",
"The population has grown by a modest percentage. The built edge has crept a short distance north and east. The halls, the river and the guilds are what they were.",
],
"10years": [
"Veyle holds roughly 1.2 million people. The Long Quays have been cut below the salt marsh, new squares have covered the north fields, and a war on the Meren has concentrated shipping and money in the city.",
"The population is about 1.2 million, up by a third. Enclosed basins have opened at the river mouth, the first gas-lamps are being tried on the Ossel bridge, and the outer wards are under construction everywhere.",
"Veyle has perhaps 1.2 million inhabitants. Substantial growth shows as new streets, new quays and a visibly larger city, though the old low wards are as crowded as they ever were.",
],
"100years": [
"Veyle holds about six million people and remains the largest city on the coast. Steam looms, a ring railway, the Ossel drain works and endless brick rows belong to this era of manufacture.",
"The smaller city of before has become a metropolis of six million. The Great Drain, the terrace railways and mile upon mile of workers' rows account for the multiplication; the salt trade is now a small part of it.",
"Industrial growth has taken Veyle from under a million to roughly six million. The built area runs far beyond the old wards, stitched together by rail, and the city is the capital of a maritime dominion.",
],
"1000years": [
"Any figure is invention upon invention. Veyle's recorded course ran on to a peak near eight million before a long decline and a subsequent recovery. Beyond that lies pure extrapolation; the delta is subsiding.",
"The city's known history occupies a small part of what could still unfold: growth to nearly eight million, a plateau and decline, then renewed growth. What stands on the Ossel further on cannot be stated.",
"The question leaves demography entirely. Veyle has already been occupied for a very long stretch; further growth, shrinkage, drowning or abandonment are all live possibilities. The rising sea and the sinking delta set the terms.",
],
"10000years": [
"There is no basis for a population figure. The Ossel delta as it now exists is the work of its own recent history and will not persist indefinitely. Any settlement on that coast would be a different city entirely.",
"The continuity of Veyle cannot be assumed. The interval considered is as long as the whole of farming on that coast. Whoever lives at the river mouth would be as far from the Veylers as the Veylers are from the first terrace builders.",
"The subject is removed. The population of Veyle at that remove is neither large nor small; the question fails: neither the city nor its language nor its shoreline can be carried that far forward.",
],
"1000000years": [
"The people who built the city have an uncertain future, and the site lies under a different sea or a different ice. No figure attaches to Veyle because there is no Veyle and no continuity with those who made it.",
"The question is geological. The coast will have been glaciated and drowned repeatedly; the Ossel has changed its course before and will again. Whatever occupies the region is unrelated to the described population.",
"The interval exceeds the span of the people who kept the Ash Reckoning. The city, its people and its records are a thin band in the sediment of one warm interval. No number can be given.",
],
},
}

# t0 (Delta t = 0) is otherwise identical to v2/v1, but six subjects' p1 paraphrase opens
# with "At first the X ..." -- a direct restatement of the t0 interval phrase ("At first,").
# Same content, restatement removed.
T0_FIXES = {
"orchard": "The orchard stands in white blossom, three hundred grafted apples on semi-dwarf stock spaced six metres apart. The ground is green and mown, the frames of the hail nets are up, and a hive sits at the end of every fourth row.",
"mayfly": "The mayfly is newly winged, pale and soft, resting on a willow leaf above the water. Its gut is non-functional and its only remaining business is to moult once more and mate; its lifespan as an adult is about twenty-four hours.",
"asteroid": "The object is a dark, potato-shaped rock of roughly 1.2 kilometres, a loosely bound rubble pile under a thin regolith. It orbits between Mars and Jupiter and turns on its axis every eleven hours.",
"river": "The river runs clear and shallow over gravel, forty metres from bank to bank, curving through pasture in long meanders. Willows overhang the cut banks and sand bars have built up on the inside of the bends.",
"real_population": "The city numbers roughly one million inhabitants at the turn of the nineteenth century. Coal smoke, river trade and handicraft workshops define it; mortality is high, and only continuous migration keeps the population rising at all.",
"fictional_population": "The city numbers about nine hundred thousand at the turn of the seventh century of the Ash Reckoning. Saltworks, looming-halls and river trade define it; the ash-yards take more than the wards bear, and the inland villages make up the difference.",
}

V3 = copy.deepcopy(V2)
V3["_note"] = ("v3 (hour-32 follow-up, docs/specs/subject_clocks_v1.md's confound list): identical to v2 "
               "except every Delta t >= 1 day state span is rewritten so it never names, numbers or "
               "paraphrases its own interval (no duration words, no 'N years/days/...' construction, "
               "no token shared with the interval phrase); t0 states, phrases, control_prompts and the "
               "far-Dt vocabulary-matching property are unchanged from v2. See "
               "scripts/time_translation_leak_check.py.")

for s, dts in NEW.items():
    for t, paras in dts.items():
        assert len(paras) == 3, (s, t, len(paras))
        V3["states"][s][t] = paras

for s, text in T0_FIXES.items():
    V3["states"][s]["t0"][1] = text

for key, marked in list(V3["prompts"].items()):
    s, t, p = key.split("/")
    pi = int(p[1:])
    interval_phrase = V2["phrases"][t]
    state_text = V3["states"][s][t][pi]
    V3["prompts"][key] = f"[[interval: {interval_phrase}]] [[state: {state_text}]]"

# control_prompts reattach each interval phrase to the subject's/paraphrase's t0 state;
# regenerate so the six T0_FIXES subjects' controls use the de-restated t0 text too.
for key in list(V3["control_prompts"].keys()):
    s, t, p = key.split("/")
    pi = int(p[1:])
    interval_phrase = V2["phrases"][t]
    t0_text = V3["states"][s]["t0"][pi]
    V3["control_prompts"][key] = f"[[interval: {interval_phrase}]] [[state: {t0_text}]]"

for key in V2["prompts"]:
    s, t, p = key.split("/")
    if t == "t0" and s not in T0_FIXES:
        assert V3["prompts"][key] == V2["prompts"][key]
        assert V3["control_prompts"][key] == V2["control_prompts"][key]

json.dump(V3, open("prompts/time_translation_v3.json", "w"), indent=1)
print("wrote prompts/time_translation_v3.json")
