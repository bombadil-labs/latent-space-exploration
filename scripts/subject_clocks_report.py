import json, sys
import numpy as np
R = json.load(open(sys.argv[1]))
L = sys.argv[2] if len(sys.argv) > 2 else "14"
S = R["_meta"]["subjects"]; DT = R["_meta"]["deltas"]; ORDER = R["_meta"]["order"]
print("== 3.1 floor ratios ||resid||/F, C1-mean, layer", L)
for s in S:
    d = R["m31_floor"][L]["per_subject"][s]
    print(f"  {s:22s}", " ".join(f"{d[t]['ratio']:5.2f}" for t in DT),
          "| F=", f"{d[DT[0]]['floor']:.1f}", "resid=", f"{d[DT[0]]['resid_norm']:.1f}")
print("  d_norm/shared l14:", {t: round(R["m31_floor"][L]["shared_norm"][t], 1) for t in DT[:3]})
for key in ["C1_mean", "C1_last", "C3_last", "C2_ilast"]:
    print(f"== 3.2/3.4 tau, {key}, layer {L}")
    for s in S:
        d = R["m32_y_tau"][key][L][s]
        print(f"  {s:22s} tau={str(d['tau']):14s} P/sig={d['P_over_sigma']:6.2f} "
              f"P={d['P']:8.2f} sig={d['sigma']:7.2f} loo={d['tau_loo']}")
print("== y means (C1_last, sigma units), layer", L)
for s in S:
    d = R["m32_y_tau"]["C1_last"][L][s]
    print(f"  {s:22s}", " ".join(f"{d['ybar'][t] / d['sigma']:6.2f}" for t in ORDER))
print("== 3.2 cross-subject diagonal dominance")
for key in R["m32_crosssubject"]:
    for lay in R["m32_crosssubject"][key]:
        row = R["m32_crosssubject"][key][lay]
        print(f"  {key} L{lay}: D=", " ".join(f"{row[t]['D']:.2f}" for t in DT),
              " p=", " ".join(f"{row[t]['p']:.3f}" for t in DT))
        print(f"  {key} L{lay}: Dc=", " ".join(f"{row[t]['D_centred']:.2f}" for t in DT),
              " pc=", " ".join(f"{row[t]['p_centred']:.3f}" for t in DT))
print("== 3.3 kappa layer", L)
for s in S:
    print(f"  {s:22s}", " ".join(f"{R['m33_kappa'][L][s][t]:6.2f}" for t in ORDER))
print("== 3.4 resolvability layer", L)
for ro in ["h_last", "y"]:
    for s in S:
        d = R["m34_resolvability"][L][ro][s]
        print(f"  {ro:7s} {s:22s} first={str(d['first_resolved_from_t0']):12s} "
              f"pred={str(d['predicted_first']):12s} agree={d['block_agreement']:.2f} "
              f"n={d['n_labelled']} fracres={d['frac_resolved']:.2f}")
print("== 3.5 discrimination layer", L)
for s in S:
    d = R["m35_discrimination"][L][s]
    print(f"  {s:22s} within={d['within_spearman']:.2f}/{d['within_mae']:.2f} "
          f"shared={d['shared_spearman']:.2f}/{d['shared_mae']:.2f} "
          f"matched={d['matched_spearman']:.2f}/{d['matched_mae']:.2f}")
print("== per-token C3 SD over dt, layer", L)
for s in S:
    v = R["m3_pertoken_C3"][L][s]["sd_over_dt"]
    print(f"  {s:22s}", " ".join(f"{x:.2f}" for x in v[:10]), "...", f"{np.mean(v[10:]):.2f}")
print("== power", json.dumps(R["m5_power"], default=float)[:600])
print("== gate", R["m7_gate"])
