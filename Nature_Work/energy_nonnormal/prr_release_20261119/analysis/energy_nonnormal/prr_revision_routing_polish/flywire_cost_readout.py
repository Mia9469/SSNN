"""Compare the pinned FlyWire gain states with no modulation, without refitting."""
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import flywire_gain_routing as fw


def window_covariance(transpose_apply, selectors, steps, prehistory):
    """Covariance of the average at steps 1..steps under unit innovations.

    Each column of weights is the adjoint response to one averaged readout.
    Sum the outer products of weights for independent innovations, including
    pre-cue history. The response to the cue does not alter this covariance.
    """
    weights = np.zeros_like(selectors)
    cov = np.zeros((selectors.shape[1], selectors.shape[1]))
    for _ in range(steps):
        weights = selectors / steps + transpose_apply(weights)
        cov += weights.T @ weights
    for _ in range(prehistory + 1):
        weights = transpose_apply(weights)
        cov += weights.T @ weights
    return cov


def main():
    # Independent scalar AR(1) check of the full lag-covariance sum.
    r, t = .7, 24
    lag = np.abs(np.arange(t)[:, None] - np.arange(t))
    np.testing.assert_allclose(window_covariance(lambda x: r*x, np.ones((1, 1)), t, 120),
                               np.mean(r**lag)/(1-r*r), rtol=1e-12)
    record = json.loads((HERE.parent / "flywire_gain_routing_results.json").read_text())
    params = record["parameters"]
    for path, key in [(fw.DEFAULT_GRAPH, "graph_sha256"), (fw.DEFAULT_NODES, "node_sha256")]:
        assert fw.sha256(path) == record["provenance"][key], path
    graph = fw.load_full_graph(fw.DEFAULT_GRAPH, fw.DEFAULT_NODES)
    n = graph["n"]
    p = graph["p_signed"]
    pt = p.T.tocsr()
    beta, gamma = params["beta"], params["gamma"]
    horizon = params["cost_steps"]
    steps = params["response_steps"]
    rng = np.random.default_rng(params["null_seed"])
    modules = {name: np.array([graph["id_to_index"][r] for r in
                               record["gain_modules"][name+"_ids"]], dtype=np.int32)
               for name in ("feeding", "grooming")}
    # Reproduce the original probe stream without repeating the null propagations.
    for _ in range(params["nulls"]):
        fw.stratified_permutation(modules["feeding"], modules["grooming"], graph,
                                  rng, bins=params["null_feature_bins"])
    probes = rng.choice(np.array([-1, 1], dtype=np.float32),
                        size=(n, params["cost_probes"]))
    rows = {}
    traces = {}
    for state in ("unmodulated", "feeding", "grooming"):
        mod = modules.get(state, np.empty(0, dtype=int))
        gain = fw.make_gain(n, mod, params["gain_high"])
        op = fw.GainPropagator(p, gain, beta, gamma)
        x = probes.copy()
        per_probe = np.sum(x*x, axis=0, dtype=np.float64) / n
        legacy = [float(np.mean(np.sum(x*x, axis=0))/n)]
        for _ in range(horizon):
            x = op.matmat(x)
            per_probe += np.sum(x*x, axis=0, dtype=np.float64) / n
            legacy.append(float(np.mean(np.sum(x*x, axis=0))/n))
        traces[state] = per_probe
        row = {"cost_per_neuron": float(per_probe.mean()), "routes": {}}
        if state != "unmodulated":
            key = "feed" if state == "feeding" else "groom"
            np.testing.assert_allclose(sum(legacy),
                                       record["global_matching"]["raw_finite_horizon_cost_"+key],
                                       rtol=5e-7)
        for task in fw.TASKS:
            _, outputs, b, _ = fw.task_vectors(task, graph)
            response = fw.route_response(op, b, outputs, steps)
            if state != "unmodulated":
                np.testing.assert_allclose(response["peak"],
                                           record["response"][task.name]["state_"+key]["peak"],
                                           rtol=2e-6)
            x = b.copy()
            mu = np.zeros(len(outputs))
            for _ in range(steps):
                x = op.matvec(x)
                mu += x[outputs] / steps
            selectors = np.zeros((n, len(outputs)))
            selectors[outputs, np.arange(len(outputs))] = 1

            def adjoint(z):
                return beta*z + gamma*(pt @ (gain[:, None]*z))

            cov = window_covariance(adjoint, selectors, steps, 40)
            converged = window_covariance(adjoint, selectors, steps, 120)
            np.testing.assert_allclose(cov, converged, atol=5e-6, rtol=1e-4)
            row["routes"][task.name] = {
                "peak_squared_transfer": response["peak"],
                "mean_squared_window_response": float(mu @ mu),
                "dprime": float(2*np.sqrt(mu @ np.linalg.solve(converged, mu))),
                "covariance_truncation_max_abs_diff": float(np.max(abs(cov-converged))),
            }
        rows[state] = row
        print(state, row, flush=True)
    base = traces["unmodulated"]
    for state in ("feeding", "grooming"):
        delta = traces[state] - base
        rows[state]["cost_change_vs_unmodulated_fraction"] = float(delta.mean()/base.mean())
        rows[state]["cost_change_paired_mc_se_fraction"] = float(
            delta.std(ddof=1)/np.sqrt(len(base))/base.mean())
    (HERE / "flywire_cost_readout_results.json").write_text(json.dumps({
        "source_graph_sha256": record["provenance"]["graph_sha256"],
        "source_node_sha256": record["provenance"]["node_sha256"],
        "cost_steps_inclusive": [0, horizon], "shared_trace_probes": len(base),
        "readout_steps_inclusive": [1, steps], "innovation_covariance": "identity",
        "cue": "positive or negative unit-normalized input indicator",
        "covariance_prehistory_steps": 120, "states": rows,
    }, indent=2)+"\n")


if __name__ == "__main__":
    main()
