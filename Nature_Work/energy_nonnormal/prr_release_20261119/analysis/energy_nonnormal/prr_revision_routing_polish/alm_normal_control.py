"""Control-trained same-spectrum normal comparison on the original ALM cohort."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm, expm_frechet, schur
from scipy.optimize import linear_sum_assignment, minimize
from scipy.stats import binomtest, wilcoxon

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import multisession_alm_validation as old


def spectrum_error(a, b):
    d = abs(np.linalg.eigvals(a)[:, None] - np.linalg.eigvals(b)[None, :])
    i, j = linear_sum_assignment(d)
    return float(d[i, j].max())


def canonical(a):
    """Real normal blocks in the original real-Schur order and handedness."""
    t, q = schur(a, output="real")
    b = np.zeros_like(t)
    i = 0
    while i < len(t):
        if i + 1 < len(t) and t[i + 1, i] != 0:
            t2 = t[i:i+2, i:i+2]
            eig = np.linalg.eigvals(t2)[0]
            re, im = eig.real, eig.imag
            w = np.copysign(abs(im), t2[1, 0])
            b[i:i+2, i:i+2] = [[re, -w], [w, re]]
            i += 2
        else:
            b[i, i] = t[i, i]
            i += 1
    return b, q


def chart(base, q0, x, y):
    g = x.T @ x / len(x)
    h = y.T @ x / len(x)
    yy = np.sum(y*y) / len(x)
    u, v = np.triu_indices(len(base), 1)

    def objective(p):
        k = np.zeros_like(base)
        k[u, v], k[v, u] = p, -p
        q = expm(k) @ q0
        a = q @ base @ q.T
        ag = a @ g
        loss = (np.sum(a * (ag - 2*h)) + yy) / len(base)
        da = 2*(ag - h) / len(base)
        dq = da @ q @ base.T + da.T @ q @ base
        dk = expm_frechet(k.T, dq @ q0.T, compute_expm=False)
        return float(loss), (dk[u, v] - dk[v, u])

    def orientation(p):
        k = np.zeros_like(base)
        k[u, v], k[v, u] = p, -p
        return expm(k) @ q0

    return objective, orientation, len(u)


def optimize(base, starts, x, y):
    best = None
    log = []
    for q0 in starts:
        f, get_q, d = chart(base, q0, x, y)
        initial = f(np.zeros(d))[0]
        r = minimize(f, np.zeros(d), jac=True, method="L-BFGS-B",
                     options={"maxiter": 450, "gtol": 2e-8, "ftol": 1e-14})
        q = get_q(r.x)
        a = q @ base @ q.T
        mse = float(np.mean((x @ a.T-y)**2))
        if not np.isfinite(mse) or abs(mse-r.fun) > 1e-8 or mse > initial+1e-9:
            raise AssertionError("orientation optimization failed objective check")
        log.append({"initial_mse": initial, "mse": mse, "success": bool(r.success),
                    "iterations": int(r.nit), "gradient_max": float(max(abs(r.jac)))})
        if best is None or mse < best[0]:
            best = mse, q
    return best[1], log


def score(a, k, x, y, covars):
    pred = x @ np.linalg.matrix_power(a, k).T
    log_pred = np.log(np.sum(pred**2, axis=1)+1e-9)
    log_obs = np.log(np.sum(y**2, axis=1)+1e-9)
    return {
        "r": old._safe_spearman(old._residualize(log_pred, covars[:, :1]),
                                old._residualize(log_obs, covars[:, :1])),
        "r_conditioned": old._safe_spearman(old._residualize(log_pred, covars),
                                            old._residualize(log_obs, covars)),
        "state_mse": float(np.mean((pred-y)**2)),
    }


def session(path, dim, seed):
    rec = old.load_session(path)
    if len(rec["units"]) < 20:
        raise ValueError("fewer than twenty stable high-quality units")
    centers, rates = old.make_trial_rates(rec)
    labels, photo = rec["labels"], rec["photo_types"]
    scores, _ = old.residual_latent_scores(rates, centers, labels, photo, dim)
    a, fit = old.fit_operator(scores, centers, photo)
    normal, q0 = canonical(a)
    rng = np.random.default_rng(seed)
    starts = [q0, np.eye(len(a))] + [np.linalg.qr(rng.normal(size=a.shape))[0]
                                     for _ in range(10)]
    control = photo == 0
    tt = np.flatnonzero((centers[:-1] >= old.FIT_WINDOW[0]) &
                         (centers[1:] < old.FIT_WINDOW[1]))
    x = scores[control][:, tt].reshape(-1, len(a))
    y = scores[control][:, tt+1].reshape(-1, len(a))
    q, log = optimize(normal, starts, x, y)
    b = q @ normal @ q.T
    k = round((np.mean(old.POST_WINDOW)-np.mean(old.EARLY_WINDOW))/old.DT)
    early = old._window_mean(scores, centers, old.EARLY_WINDOW)
    late = old._window_mean(scores, centers, old.POST_WINDOW)
    # Sensitivity gives the normal comparator access to the actual evaluation
    # windows on controls, still never selecting an orientation on inhibition.
    q_h, log_h = optimize(np.linalg.matrix_power(normal, k), [q]+starts,
                          early[control], late[control])
    b_h = q_h @ normal @ q_h.T
    for matrix in (b, b_h):
        if spectrum_error(matrix, a) > 1e-7:
            raise AssertionError("normal spectrum not matched")
        if np.linalg.norm(matrix.T @ matrix-matrix @ matrix.T) > 1e-10:
            raise AssertionError("comparison matrix not normal")
    test = np.isin(photo, [1, 2, 3])
    xt, yt = early[test], late[test]
    covars = np.c_[np.log(np.sum(xt**2, axis=1)+1e-9), photo[test]==2,
                   photo[test]==3, labels[test]>0]
    return {"session": rec["session"], "mouse": rec["mouse"], "file": path.name,
            "fit": fit, "n_inhibition": int(test.sum()), "steps": k,
            "dimension": len(a), "spectral_radius": float(max(abs(np.linalg.eigvals(a)))),
            "full": score(a, k, xt, yt, covars),
            "normal": score(b, k, xt, yt, covars),
            "normal_window_fit": score(b_h, k, xt, yt, covars),
            "full_train_mse": float(np.mean((x @ a.T-y)**2)),
            "normal_train_starts": log, "normal_window_starts": log_h,
            "spectrum_max_error": max(spectrum_error(b, a), spectrum_error(b_h, a)),
            "normality_max_error": max(float(np.linalg.norm(m.T@m-m@m.T)) for m in (b,b_h))}


def summarize(reports):
    mice = sorted({r["mouse"] for r in reports})
    by_mouse = {}
    for m in mice:
        rr = [r for r in reports if r["mouse"] == m]
        by_mouse[m] = {key: {metric: float(np.mean([r[key][metric] for r in rr]))
                             for metric in ("r", "r_conditioned", "state_mse")}
                       for key in ("full", "normal", "normal_window_fit")}
    summary = {}
    for key in ("normal", "normal_window_fit"):
        summary[key] = {}
        for metric in ("r", "r_conditioned", "state_mse"):
            direction = -1 if metric == "state_mse" else 1
            d = np.array([direction*(v["full"][metric]-v[key][metric])
                          for v in by_mouse.values()])
            if not np.isfinite(d).all():
                raise AssertionError("nonfinite mouse result")
            wins = int(np.sum(d > 0))
            n = int(np.sum(d != 0))
            summary[key][metric] = {
                "delta_full_advantage_by_mouse": d.tolist(), "positive": wins,
                "median": float(np.median(d)), "two_sided_sign_p": binomtest(wins,n).pvalue if n else 1.,
                "two_sided_wilcoxon_p": wilcoxon(d).pvalue if n else 1.,
            }
    return by_mouse, summary


def selftest():
    rng = np.random.default_rng(91)
    q = np.linalg.qr(rng.normal(size=(5,5)))[0]
    b = np.diag([.1,.2,.45,.6,.9])
    b[0:2,0:2] = [[.35,-.22],[.22,.35]]
    x = rng.normal(size=(500,5))
    target = q @ b @ q.T
    y = x @ target.T
    f, _, d = chart(b, np.eye(5), x, y)
    p = rng.normal(size=d)/3
    v = rng.normal(size=d)
    numeric = (f(p+1e-6*v)[0]-f(p-1e-6*v)[0])/2e-6
    assert abs(numeric-f(p)[1]@v) < 1e-8
    qfit, _ = optimize(b, [np.eye(5),np.linalg.qr(rng.normal(size=(5,5)))[0]], x, y)
    assert np.mean((x @ (qfit @ b @ qfit.T).T-y)**2) < 1e-12


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--dimension", type=int, default=12)
    args = parser.parse_args()
    selftest()
    previous = json.loads((HERE.parent/"alm_multisession_results.json").read_text())
    included = {r["file"] for r in previous["sessions"]}
    excluded = {r["file"] for r in previous["excluded"]}
    paths = sorted(p for p in args.data.glob("data_structure_*.mat") if not p.name.startswith("."))
    if {p.name for p in paths} != included | excluded:
        raise ValueError("files do not match the original 18-file cohort")
    reports = []
    for i,p in enumerate(paths):
        if p.name not in included:
            continue
        r = session(p,args.dimension,1291+i)
        reports.append(r)
        print(r["session"], {name: round(r[name]["r"],4) for name in
                              ("full","normal","normal_window_fit")}, flush=True)
    by_mouse, summary = summarize(reports)
    output = {"dimension": args.dimension, "original_exclusions": previous["excluded"],
              "mice": by_mouse, "summary": summary, "sessions": reports}
    (HERE/f"alm_normal_d{args.dimension}.json").write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == "__main__":
    main()
