#!/usr/bin/env python3
"""Sensitivity checks for the full FlyWire v783 gain-routing experiment."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import flywire_gain_routing as fw


HERE = Path(__file__).resolve().parent


def prepare(graph, corridor_hops: int, damping: float):
    task_cache = {}
    corridor = {}
    excluded = set()
    for task in fw.TASKS:
        score, inputs, outputs = fw.corridor_score(graph, task, corridor_hops, damping)
        task_cache[task.name] = {
            "inputs": inputs,
            "outputs": outputs,
            "b": fw.unit_set_vector(inputs, graph["n"]),
        }
        corridor[task.name] = score
        excluded.update(inputs.tolist())
        excluded.update(outputs.tolist())
    for index in excluded:
        corridor["feeding"][index] = 0
        corridor["grooming"][index] = 0
    return corridor, task_cache


def evaluate_condition(
    graph,
    corridor,
    task_cache,
    module_size: int,
    gain_high: float,
    beta: float,
    corridor_hops: int,
    damping: float,
    ranking_rule: str,
    args,
    seed: int,
):
    feed, groom = fw.choose_route_specific_modules(
        corridor["feeding"], corridor["grooming"], module_size,
        ranking_rule=ranking_rule, epsilon=args.ranking_epsilon,
    )
    parameters = {
        "gain_high": gain_high,
        "beta": beta,
        "gamma": (args.stability_bound - beta) / gain_high,
        "stability_bound": args.stability_bound,
        "response_steps": args.response_steps,
    }
    response, op_feed, op_groom = fw.evaluate_pair(
        graph, feed, groom, parameters, task_cache
    )
    switch = fw.switch_metrics(response)
    rng = np.random.default_rng(seed)
    null_switch = []
    for _ in range(args.nulls):
        null_feed, null_groom = fw.stratified_permutation(
            feed,
            groom,
            graph,
            rng,
            bins=args.null_feature_bins,
        )
        null_response, _, _ = fw.evaluate_pair(
            graph, null_feed, null_groom, parameters, task_cache
        )
        null_switch.append(
            fw.switch_metrics(null_response)["geometric_switch_factor"]
        )
    null_switch = np.asarray(null_switch)

    probes = rng.choice(
        np.array([-1.0, 1.0], dtype=np.float32),
        size=(graph["n"], args.cost_probes),
    )
    cost_feed, _ = fw.finite_horizon_cost(op_feed, probes, args.cost_steps)
    cost_groom, _ = fw.finite_horizon_cost(op_groom, probes, args.cost_steps)
    p_value = float(
        (1 + np.sum(null_switch >= switch["geometric_switch_factor"]))
        / (args.nulls + 1)
    )
    return {
        "module_size": module_size,
        "gain_high": gain_high,
        "beta": beta,
        "corridor_hops": corridor_hops,
        "corridor_damping": damping,
        "ranking_rule": ranking_rule,
        "gamma": parameters["gamma"],
        "feeding_selectivity": switch["feeding_selectivity"],
        "grooming_selectivity": switch["grooming_selectivity"],
        "geometric_switch_factor": switch["geometric_switch_factor"],
        "null_median": float(np.median(null_switch)),
        "null_q95": float(np.quantile(null_switch, 0.95)),
        "p_one_sided": p_value,
        "cost_relative_difference": float(
            abs(cost_feed - cost_groom) / max(cost_feed, cost_groom)
        ),
        "passes": bool(
            switch["feeding_selectivity"] > 1
            and switch["grooming_selectivity"] > 1
            and switch["geometric_switch_factor"] > np.quantile(null_switch, 0.95)
            and abs(cost_feed - cost_groom) / max(cost_feed, cost_groom) < 0.05
        ),
    }


def plot(results, output: Path):
    red = "#C62828"
    grey = "#90A4AE"
    plt.rcParams.update({
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.3,
        "ytick.labelsize": 7.3,
        "axes.linewidth": 0.9,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })
    fig, axes = plt.subplots(2, 3, figsize=(7.3, 4.2), constrained_layout=True)

    central = {
        "module_size": 256,
        "gain_high": 2.0,
        "beta": 0.70,
        "corridor_hops": 8,
        "corridor_damping": 0.85,
        "ranking_rule": "contrast_sqrt",
    }

    def subset_for(field, order=None):
        subset = [
            row for row in results
            if all(row[key] == value for key, value in central.items() if key != field)
        ]
        if order is None:
            return sorted(subset, key=lambda row: row[field])
        rank = {value: index for index, value in enumerate(order)}
        return sorted(subset, key=lambda row: rank[row[field]])

    ranking_order = ["difference", "contrast_sqrt", "log_ratio_sqrt"]
    specs = [
        ("module_size", "gain-module size", "(a) Module size", None),
        ("gain_high", "boosted neuronal gain", "(b) Gain amplitude", None),
        ("beta", r"self-memory $\beta$", "(c) Propagator balance", None),
        ("corridor_hops", "communicability horizon $H$", "(d) Corridor horizon", None),
        ("corridor_damping", r"communicability damping $\delta$", "(e) Corridor damping", None),
        ("ranking_rule", "corridor-ranking rule", "(f) Ranking rule", ranking_order),
    ]
    for ax, (field, xlabel, title, order) in zip(axes.ravel(), specs):
        subset = subset_for(field, order)
        if order is None:
            x = np.asarray([row[field] for row in subset], dtype=float)
        else:
            x = np.arange(len(subset), dtype=float)
        observed = np.asarray([row["geometric_switch_factor"] for row in subset])
        null95 = np.asarray([row["null_q95"] for row in subset])
        ax.plot(x, observed, "o-", color=red, label="route-defined")
        ax.plot(x, null95, "s--", color=grey, label="95% null")
        ax.axhline(1, color="0.35", lw=0.8, ls=":")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("routing switch factor")
        ax.set_title(title)
        if order is not None:
            ax.set_xticks(x)
            ax.set_xticklabels(["difference", r"contrast$\sqrt{s}$", r"log-ratio$\sqrt{s}$"], rotation=20)
        ax.legend(frameon=False, fontsize=6.4)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=fw.DEFAULT_GRAPH)
    parser.add_argument("--nodes", type=Path, default=fw.DEFAULT_NODES)
    parser.add_argument("--nulls", type=int, default=99)
    parser.add_argument("--null-feature-bins", type=int, default=2)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--stability-bound", type=float, default=0.98)
    parser.add_argument("--ranking-epsilon", type=float, default=1e-20)
    parser.add_argument("--response-steps", type=int, default=24)
    parser.add_argument("--cost-probes", type=int, default=8)
    parser.add_argument("--cost-steps", type=int, default=30)
    parser.add_argument("--output", type=Path, default=HERE / "flywire_gain_routing_sensitivity.json")
    parser.add_argument("--figure", type=Path, default=HERE / "figures" / "fig_flywire_gain_routing_sensitivity.pdf")
    args = parser.parse_args()

    started = time.time()
    graph = fw.load_full_graph(args.graph, args.nodes)
    central = (256, 2.0, 0.70, 8, 0.85, "contrast_sqrt")
    conditions = set()
    conditions.update((size, 2.0, 0.70, 8, 0.85, "contrast_sqrt")
                      for size in (128, 256, 512))
    conditions.update((256, gain, 0.70, 8, 0.85, "contrast_sqrt")
                      for gain in (1.5, 2.0, 2.5))
    conditions.update((256, 2.0, beta, 8, 0.85, "contrast_sqrt")
                      for beta in (0.60, 0.70, 0.80))
    conditions.update((256, 2.0, 0.70, hops, 0.85, "contrast_sqrt")
                      for hops in (4, 8, 12))
    conditions.update((256, 2.0, 0.70, 8, damping, "contrast_sqrt")
                      for damping in (0.65, 0.85, 1.0))
    conditions.update((256, 2.0, 0.70, 8, 0.85, ranking)
                      for ranking in ("difference", "contrast_sqrt", "log_ratio_sqrt"))
    conditions = sorted(conditions, key=lambda row: tuple(str(value) for value in row))
    corridor_cache = {}
    results = []
    for index, (module_size, gain_high, beta, hops, damping, ranking_rule) in enumerate(conditions):
        cache_key = (hops, damping)
        if cache_key not in corridor_cache:
            corridor_cache[cache_key] = prepare(graph, hops, damping)
        corridor, task_cache = corridor_cache[cache_key]
        result = evaluate_condition(
            graph, corridor, task_cache, module_size, gain_high, beta, hops,
            damping, ranking_rule, args,
            seed=args.seed + index,
        )
        results.append(result)
        print(json.dumps(result))

    central_corridor, _ = corridor_cache[(central[3], central[4])]
    epsilon_modules = {}
    reference_modules = None
    for epsilon in (1e-12, 1e-20, 1e-30):
        modules = fw.choose_route_specific_modules(
            central_corridor["feeding"], central_corridor["grooming"], 256,
            ranking_rule="contrast_sqrt", epsilon=epsilon,
        )
        if reference_modules is None:
            reference_modules = modules
        epsilon_modules[f"{epsilon:.0e}"] = bool(
            np.array_equal(modules[0], reference_modules[0])
            and np.array_equal(modules[1], reference_modules[1])
        )

    summary = {
        "provenance": {
            "source_commit": fw.SOURCE_COMMIT,
            "source_doi": fw.SOURCE_DOI,
            "connectome_doi": fw.CONNECTOME_DOI,
            "materialization_version": fw.SOURCE_VERSION,
            "graph_sha256": fw.sha256(args.graph),
            "node_sha256": fw.sha256(args.nodes),
            "graph": {
                "neurons": graph["n"],
                "edges": graph["m"],
            },
            "root_id_mappings": {
                str(old): {
                    "v783_root": int(new),
                    "stable_supervoxel": int(supervoxel),
                }
                for old, (new, supervoxel) in fw.ROOT_ID_MAPPINGS.items()
            },
        },
        "parameters": {
            "nulls_per_condition": args.nulls,
            "null_feature_bins": args.null_feature_bins,
            "seed": args.seed,
            "common_stability_bound": args.stability_bound,
            "central_condition": {
                "module_size": central[0], "gain_high": central[1],
                "beta": central[2], "corridor_hops": central[3],
                "corridor_damping": central[4], "ranking_rule": central[5],
            },
            "ranking_epsilon": args.ranking_epsilon,
            "cost_probes": args.cost_probes,
            "cost_steps": args.cost_steps,
        },
        "conditions": results,
        "all_conditions_pass": bool(all(row["passes"] for row in results)),
        "epsilon_invariance": epsilon_modules,
        "runtime_seconds": float(time.time() - started),
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    plot(results, args.figure)
    print(json.dumps({
        "all_conditions_pass": summary["all_conditions_pass"],
        "runtime_seconds": summary["runtime_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
