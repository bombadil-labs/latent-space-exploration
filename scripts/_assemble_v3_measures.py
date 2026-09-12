import json

v3_measures = json.load(open("results/time_translation_v3_fresh_measures.json"))
v2_measures = json.load(open("results/time_translation_v2_fresh_measures.json"))
disc_v2 = json.load(open("results/time_translation_v2_discrimination.json"))
disc_v3 = json.load(open("results/time_translation_v3_discrimination.json"))
shared_cos = json.load(open("results/time_translation_v2_v3_shared_cos.json"))

out = {
    "_meta": {
        "description": "v3 time-translation measures (state spans rewritten to remove lexical "
                        "restatement of the interval), plus v2-vs-v3 comparison artifacts. "
                        "The full per-Dt/per-layer measure suite (m1-m7) is in "
                        "results/time_translation_v3_fresh_measures.json (identical script, "
                        "unchanged, run on prompts/time_translation_v3.json); this file adds the "
                        "cross-grid comparisons requested for the v3 report.",
        "v2_measures_file": "results/time_translation_v2_fresh_measures.json",
        "v3_measures_file": "results/time_translation_v3_fresh_measures.json",
    },
    "v3_m1_m2_decomposition": v3_measures["m1_m2_decomposition"],
    "v3_m3_clock_geometry": v3_measures["m3_clock_geometry"],
    "v3_m7_phrase_control": v3_measures["m7_phrase_control"],
    "v2_m1_m2_decomposition": v2_measures["m1_m2_decomposition"],
    "v2_m3_clock_geometry": v2_measures["m3_clock_geometry"],
    "v2_m7_phrase_control": v2_measures["m7_phrase_control"],
    "discrimination_sec3_5": {"v2": disc_v2, "v3": disc_v3},
    "cos_shared_v2_v3": shared_cos,
}
json.dump(out, open("results/time_translation_v3_measures.json", "w"), indent=1)
print("wrote results/time_translation_v3_measures.json")
