#!/usr/bin/env python
"""
Unified TCC (points 1+2). Does NOT touch existing results; writes tcc_fused_*.json only.

Pipeline:
  s_fused = z(MSP-M2F) + beta * z(cDNP)          # beta=1 main (ID-only, no OOD-tuning)
  TCC applied to s_fused (NOT raw MSP-M2F):
    - z-norm stats: validation ID pixels only (per dataset, global scalar)
    - calibration: per-image per-predicted-class subsample of val-ID pixels (cap), pooled
    - finite-sample split-conformal quantile: k_c = ceil((n_c+1)(1-alpha)),
        tau_c = order-statistic s^c_(k_c);  n_c<100 -> global finite-sample threshold
    - flag pixel OOD iff s_fused_i > tau_{chat_i}

Self-checks (must pass before trusting new numbers):
  (A) after z-norm, val-ID mean~0, std~1 for both scores
  (B) AP(beta=0 fused) == cached MSP-M2F AP (z-norm is monotone) ; AP(beta=tuned) ~ method_final json Ours AP
  (C) reproduce conformal_headroom (raw MSP-M2F, np.percentile, no subsample) FPR/recall
"""
import argparse, json, numpy as np
from collections import defaultdict
from ontology import IGNORE, ROOT
from datasets import get_dataset
import metrics as M

R = f"{ROOT}/results"
DS_LIST = ["goose", "rellis", "wildscenes", "rugd"]


def load(ds, split, seed):
    pfx = "" if ds == "goose" else f"{ds}_"
    ssfx = "" if seed == 0 else f"_s{seed}"
    pc = np.load(f"{R}/m2f_predclass_{pfx}{split}{ssfx}.npz")
    cd = np.load(f"{R}/cdnp_scores_{pfx}{split}.npz")   # cDNP is frozen-DINOv2 -> seed-independent
    assert np.array_equal(pc["keys"], cd["keys"]), f"{ds}/{split} keys misaligned (predclass vs cdnp)"
    return (pc["predclass"].astype(np.int32), pc["MSP_M2F"].astype(np.float32),
            cd["scores"].astype(np.float32), pc["keys"])          # each [F,H,W]


def zstats(x_flat, idmask_flat):
    v = x_flat[idmask_flat]
    return float(v.mean()), float(v.std() + 1e-6)


def finite_sample_quantile(s1d, alpha):
    n = len(s1d)
    if n == 0:
        return np.inf
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.inf                      # cannot guarantee -> flag nothing (FPR<=alpha)
    return float(np.partition(s1d, k - 1)[k - 1])   # k-th smallest (1-indexed)


def subsample_calib(fused, predclass, idmask, cap, rng):
    """per-image per-predicted-class subsample of ID-pixel fused scores -> dict c->array."""
    per_c = defaultdict(list)
    F = fused.shape[0]
    for i in range(F):
        m = idmask[i]
        if not m.any():
            continue
        pcs = predclass[i][m]; fus = fused[i][m]
        for c in np.unique(pcs):
            fc = fus[pcs == c]
            if cap is not None and len(fc) > cap:
                fc = rng.choice(fc, cap, replace=False)
            per_c[int(c)].append(fc)
    return {c: np.concatenate(v) for c, v in per_c.items()}


def run(dataset, seeds, alpha=0.05, cap=200, verbose=True):
    _, ol = get_dataset(dataset); o = ol()
    tl = np.full(256, IGNORE, np.uint8)
    for k, v in o["test_lut"].items():
        tl[k] = v

    # tuned beta (oracle) from existing method_final json, for self-check B
    pfx = "" if dataset == "goose" else f"{dataset}_"
    try:
        mf = json.load(open(f"{R}/method_final_MSP-M2F_{pfx}test.json"))
        tuned_beta = float(mf.get("beta", 1.0)); tuned_ap_ref = mf.get("AP", mf.get("our_AP"))
    except Exception:
        tuned_beta, tuned_ap_ref = 1.0, None

    out = {"dataset": dataset, "alpha": alpha, "cap": cap, "seeds": list(seeds)}
    per_seed_op = {"rawMSP_tcc_finite": [], "fused_tcc_finite": [], "raw_tcc_percentile_REPRO": [],
                   "global_conformal": []}
    per_seed_unif = []          # per-terrain-class test-FPR uniformity: global vs Ours-TCC
    MIN_CLS_PX = 2000           # only score predicted-terrain classes with enough test ID pixels
    ap_seed0 = {}

    for si, seed in enumerate(seeds):
        pcv, mv, cv, kv = load(dataset, "val", seed)
        pct, mt, ct, kt = load(dataset, "test", seed)
        gv = tl[kv]; gt = tl[kt]
        idv = (gv == 0); oodv = (gv == 1)
        idt = (gt == 0); oodt = (gt == 1)

        mvf, cvf = mv.ravel(), cv.ravel()
        mtf, ctf = mt.ravel(), ct.ravel()
        idvf, oodvf = idv.ravel(), oodv.ravel()
        idtf, oodtf = idt.ravel(), oodt.ravel()

        # --- z-norm stats on val-ID ---
        muM, sdM = zstats(mvf, idvf); muC, sdC = zstats(cvf, idvf)
        zMv = (mvf - muM) / sdM; zCv = (cvf - muC) / sdC
        zMt = (mtf - muM) / sdM; zCt = (ctf - muC) / sdC

        # (A) z-norm self-check
        if verbose and si == 0:
            print(f"  [A z-norm] val-ID zMSP mean={zMv[idvf].mean():+.3f} std={zMv[idvf].std():.3f} | "
                  f"zcDNP mean={zCv[idvf].mean():+.3f} std={zCv[idvf].std():.3f}  (expect ~0, ~1)")

        def fused(zM, zC, beta):
            return zM + beta * zC

        def ap_of(flat_scores):
            # reuse the paper's pixel AP on test (single concatenated frame is fine)
            return M.pixel_pr_auroc_fpr95(flat_scores.reshape(1, -1), gt.reshape(1, -1))

        # (B) AP self-checks on seed 0
        if si == 0:
            ap_b0 = ap_of(fused(zMt, zCt, 0.0))          # == MSP-M2F only (monotone)
            ap_b1 = ap_of(fused(zMt, zCt, 1.0))
            ap_bt = ap_of(fused(zMt, zCt, tuned_beta))
            ap_seed0 = {"beta0_MSPM2F": ap_b0, "beta1_MAIN": ap_b1, f"beta_tuned({tuned_beta})": ap_bt}
            if verbose:
                print(f"  [B AP] beta=0(=MSP-M2F) AP={ap_b0['AP']:.3f} FPR95={ap_b0['FPR95']:.3f} | "
                      f"beta=1 AP={ap_b1['AP']:.3f} FPR95={ap_b1['FPR95']:.3f} | "
                      f"beta_tuned={tuned_beta} AP={ap_bt['AP']:.3f}"
                      + (f"  (method_final ref AP={tuned_ap_ref})" if tuned_ap_ref else ""))

        # === MAIN: fused (beta=1) + finite-sample per-image-subsampled TCC ===
        beta = 1.0
        sfv = fused(zMv, zCv, beta); sft = fused(zMt, zCt, beta)
        sfv3 = sfv.reshape(mv.shape)   # back to [F,H,W] for per-image subsample
        rng = np.random.default_rng(seed)
        calib = subsample_calib(sfv3, pcv, idv, cap, rng)
        tg = finite_sample_quantile(sfv[idvf], alpha)          # global finite-sample
        tau_c = {c: (finite_sample_quantile(s, alpha) if len(s) >= 100 else tg) for c, s in calib.items()}
        maxc = int(max(pcv.max(), pct.max())) + 1
        tau_map = np.full(maxc, tg, np.float32)
        for c, t in tau_c.items():
            if c < maxc:
                tau_map[c] = t
        thr_t = tau_map[np.clip(pct.ravel(), 0, maxc - 1)]
        flag = sft > thr_t
        fpr = float((flag & idtf).sum() / max(idtf.sum(), 1))
        rec = float((flag & oodtf).sum() / max(oodtf.sum(), 1))
        per_seed_op["fused_tcc_finite"].append(dict(FPR=fpr, recall=rec))

        # === Ours-TCC (deployment): raw MSP-M2F + finite-sample + per-image subsample ===
        rngm = np.random.default_rng(1000 + seed)
        calib_m = subsample_calib(mv, pcv, idv, cap, rngm)     # mv is [F,H,W] raw MSP-M2F
        tgm = finite_sample_quantile(mvf[idvf], alpha)
        tau_cm = {c: (finite_sample_quantile(s, alpha) if len(s) >= 100 else tgm) for c, s in calib_m.items()}
        tmap_m = np.full(maxc, tgm, np.float32)
        for c, t in tau_cm.items():
            if c < maxc:
                tmap_m[c] = t
        thr_m = tmap_m[np.clip(pct.ravel(), 0, maxc - 1)]
        flag_m = mtf > thr_m
        fpr_m = float((flag_m & idtf).sum() / max(idtf.sum(), 1))
        rec_m = float((flag_m & oodtf).sum() / max(oodtf.sum(), 1))
        per_seed_op["rawMSP_tcc_finite"].append(dict(FPR=fpr_m, recall=rec_m))

        # === Global conformal (MSP-M2F, val-ID): single pooled finite-sample quantile, NO terrain conditioning ===
        # Same score (raw MSP-M2F), same ID-only finite-sample split-conformal target alpha, but ONE global
        # threshold tgm applied to every pixel regardless of predicted terrain -> isolates the effect of the
        # terrain conditioning in Ours-TCC (tgm is exactly the per-terrain global fallback used above).
        flag_g = mtf > tgm
        fpr_g = float((flag_g & idtf).sum() / max(idtf.sum(), 1))
        rec_g = float((flag_g & oodtf).sum() / max(oodtf.sum(), 1))
        per_seed_op["global_conformal"].append(dict(FPR=fpr_g, recall=rec_g))

        # === Per-terrain FPR uniformity: for each predicted terrain class, the test ID FPR under the
        # global threshold vs the Ours-TCC per-class threshold. A global threshold controls the MARGINAL
        # FPR but leaves some terrain classes far above target; conditioning targets alpha per class. ===
        pctf = pct.ravel()
        fprc_g, fprc_t = [], []
        for c in np.unique(pctf[idtf]):
            idc = idtf & (pctf == c); n_c = int(idc.sum())
            if n_c < MIN_CLS_PX:
                continue
            fprc_g.append(float((flag_g & idc).sum() / n_c))   # global threshold, this terrain
            fprc_t.append(float((flag_m & idc).sum() / n_c))   # Ours-TCC threshold, this terrain
        fprc_g = np.array(fprc_g); fprc_t = np.array(fprc_t)
        bad = 2 * alpha                                        # "badly miscalibrated": FPR > 2*alpha
        per_seed_unif.append(dict(
            n_classes=int(len(fprc_g)),
            worst_global=float(fprc_g.max()), worst_tcc=float(fprc_t.max()),
            frac_bad_global=float((fprc_g > bad).mean()), frac_bad_tcc=float((fprc_t > bad).mean()),
            mad_global=float(np.abs(fprc_g - alpha).mean()), mad_tcc=float(np.abs(fprc_t - alpha).mean())))

        # === (C) reproduce conformal_headroom: raw MSP-M2F, np.percentile, no subsample ===
        pcvf = pcv.ravel()
        classes = np.unique(pcvf[idvf])
        tg_r = np.percentile(mvf[idvf], 100 * (1 - alpha))
        tau_r = {}
        for c in classes:
            m = idvf & (pcvf == c); sc = mvf[m]
            tau_r[c] = float(np.percentile(sc, 100 * (1 - alpha))) if sc.size >= 100 else tg_r
        tmap_r = np.full(maxc, tg_r, np.float32)
        for c, t in tau_r.items():
            if c < maxc:
                tmap_r[c] = t
        thr_r = tmap_r[np.clip(pct.ravel(), 0, maxc - 1)]
        flag_r = mtf > thr_r
        fpr_r = float((flag_r & idtf).sum() / max(idtf.sum(), 1))
        rec_r = float((flag_r & oodtf).sum() / max(oodtf.sum(), 1))
        per_seed_op["raw_tcc_percentile_REPRO"].append(dict(FPR=fpr_r, recall=rec_r))

    # aggregate operating points over seeds
    def agg(rs):
        fpr = np.array([r["FPR"] for r in rs]); rec = np.array([r["recall"] for r in rs])
        return dict(FPR_mean=float(fpr.mean()), FPR_std=float(fpr.std()), FPR_cv=float(fpr.std() / (fpr.mean() + 1e-9)),
                    recall_mean=float(rec.mean()), recall_std=float(rec.std()),
                    per_seed_FPR=fpr.tolist(), per_seed_recall=rec.tolist())
    out["op_rawMSP_tcc_finite(DEPLOY)"] = agg(per_seed_op["rawMSP_tcc_finite"])
    out["op_fused_tcc_finite(ABLATION)"] = agg(per_seed_op["fused_tcc_finite"])
    out["op_raw_tcc_percentile_REPRO"] = agg(per_seed_op["raw_tcc_percentile_REPRO"])
    out["op_global_conformal(ABLATION)"] = agg(per_seed_op["global_conformal"])
    out["ap_seed0"] = {k: {kk: (None if kk == 'n_pos' else round(vv, 4)) for kk, vv in v.items()} for k, v in ap_seed0.items()}

    # aggregate per-terrain FPR uniformity over seeds (mean)
    def uagg(key):
        return float(np.mean([u[key] for u in per_seed_unif]))
    out["perterrain_fpr_uniformity"] = {
        "min_class_px": MIN_CLS_PX, "bad_thresh": 2 * alpha,
        "n_classes": uagg("n_classes"),
        "worst_class_FPR_global": uagg("worst_global"), "worst_class_FPR_tcc": uagg("worst_tcc"),
        "frac_classes_over_2alpha_global": uagg("frac_bad_global"), "frac_classes_over_2alpha_tcc": uagg("frac_bad_tcc"),
        "mean_abs_dev_from_alpha_global": uagg("mad_global"), "mean_abs_dev_from_alpha_tcc": uagg("mad_tcc")}

    if verbose:
        d = out["op_rawMSP_tcc_finite(DEPLOY)"]; m = out["op_fused_tcc_finite(ABLATION)"]; r = out["op_raw_tcc_percentile_REPRO"]
        g = out["op_global_conformal(ABLATION)"]
        print(f"  [C REPRO raw-percentile(old)] FPR={r['FPR_mean']:.3f}±{r['FPR_std']:.3f} recall={r['recall_mean']:.3f}  "
              f"(compare to conformal_headroom_{pfx}.json per-terrain)")
        print(f"  [GLOBAL conformal (no terrain)] FPR={g['FPR_mean']:.3f}±{g['FPR_std']:.3f} cv={g['FPR_cv']*100:.0f}% recall={g['recall_mean']:.3f}   <-- ID-only, no conditioning")
        print(f"  [DEPLOY raw-MSP + finite]     FPR={d['FPR_mean']:.3f}±{d['FPR_std']:.3f} recall={d['recall_mean']:.3f}   <-- new Ours-TCC (Table conformal)")
        print(f"  [ABLATION fused + finite]     FPR={m['FPR_mean']:.3f}±{m['FPR_std']:.3f} recall={m['recall_mean']:.3f}   <-- ranking>calibratability tradeoff")
        u = out["perterrain_fpr_uniformity"]
        print(f"  [PER-TERRAIN uniformity, {u['n_classes']:.0f} classes >= {MIN_CLS_PX}px]  "
              f"worst-class FPR: global={u['worst_class_FPR_global']:.3f} vs TCC={u['worst_class_FPR_tcc']:.3f} | "
              f"frac classes >2a: global={u['frac_classes_over_2alpha_global']*100:.0f}% vs TCC={u['frac_classes_over_2alpha_tcc']*100:.0f}% | "
              f"MAD(a): global={u['mean_abs_dev_from_alpha_global']:.3f} vs TCC={u['mean_abs_dev_from_alpha_tcc']:.3f}   <-- terrain conditioning value")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--cap", type=int, default=200)
    a = ap.parse_args()
    SEEDS = {"goose": (0, 1, 2), "rellis": (0, 1, 2), "wildscenes": (0, 1), "rugd": (0, 1)}
    datasets = DS_LIST if a.dataset == "all" else [a.dataset]
    allout = {}
    for ds in datasets:
        print(f"\n=== {ds} (alpha={a.alpha}, cap={a.cap}) ===")
        allout[ds] = run(ds, SEEDS[ds], a.alpha, a.cap)
    json.dump(allout, open(f"{R}/tcc_fused_summary.json", "w"), indent=2)
    print(f"\n[saved] results/tcc_fused_summary.json")
