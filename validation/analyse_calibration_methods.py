import json, numpy as np
rows = [json.loads(l) for l in open("calibration_methods_results.jsonl")]
TRUTH = {"tau_1_ns": 0.8, "tau_2_ns": 3.2, "amplitude_1_percent": 60.0}
print(f"{len(rows)} realisations x 4 samples\n")
for method in ("same_clip", "cv_free", "cv_clip"):
    irf_err = np.array([r[method]["irf_error"] for r in rows]) * 100
    tau_ref = np.array([r[method]["tau_ref"] for r in rows]) - 1.87
    edge = np.mean([r[method]["edge"] for r in rows])
    print(f"== {method}: IRF error {irf_err.mean():.2f} ± {irf_err.std(ddof=1)/np.sqrt(irf_err.size):.2f} %   "
          f"calibration tau bias {tau_ref.mean()*1e3:+.1f} ± {tau_ref.std(ddof=1)/np.sqrt(tau_ref.size)*1e3:.1f} ps   grid edge {edge*100:.0f} %")
    for p, truth in TRUTH.items():
        values = np.array([[s[p]["value"] for s in r[method]["samples"]] for r in rows])        # realisation x sample
        stat = np.array([[s[p]["stat"] for s in r[method]["samples"]] for r in rows])
        irf = np.array([[s[p]["irf"] for s in r[method]["samples"]] for r in rows])
        total = np.hypot(stat, irf)
        dev = values - truth
        per_cal = dev.mean(axis=1)                                   # shared (calibration) part of the error
        print(f"   {p:20s} bias {dev.mean():+.4f} ± {per_cal.std(ddof=1)/np.sqrt(per_cal.size):.4f}  | "
              f"1σ coverage: Fisher only {np.mean(np.abs(dev) < stat)*100:3.0f} %, Fisher+jackknife {np.mean(np.abs(dev) < total)*100:3.0f} %  | "
              f"rms z: {np.sqrt(np.mean((dev/stat)**2)):.2f} -> {np.sqrt(np.mean((dev/total)**2)):.2f}  | "
              f"scatter of calibration-shared error {per_cal.std(ddof=1):.4f} vs mean jackknife {irf.mean():.4f}")
    print()
