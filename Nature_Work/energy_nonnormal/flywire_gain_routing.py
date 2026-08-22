#!/usr/bin/env python3
"""Connectome-constrained gain routing on the published FlyWire v783 graph.

This analysis is a robustness/generalization test of the exact construction in
Fig. 1, not a claim that gain states were measured in vivo.  It keeps all
139,255 published neurons and the full 15.1-million-edge Shiu v783 model graph
fixed and asks whether two
predefined sensorimotor routes can be selectively reweighted by sparse,
state-dependent postsynaptic gain.

Task definitions are copied from the official Shiu et al. reproduction notebook
(`figures.ipynb`, GitHub commit 91bdd1e7dcf193f3e7ca5a8933497fcef63b7960):

* feeding: right-hemisphere sugar GRNs -> bilateral MN9;
* grooming: all published JON mechanosensory subtypes -> DN1/DN2 descending
  outputs.

Three v630 root IDs changed before materialization 783.  They are mapped by
stable supervoxel IDs in the public Codex v630/v783 label tables; the exact
old/new pairs are stored in the output provenance.

The gain modules are selected before task propagation from absolute structural
communicability between each published input/output pair.  Both states boost the
same number of neurons by the same amount.  A stratified label-permutation null
preserves module size, gain histogram, neurotransmitter sign, and joint strata
formed by separately quantiling raw out-strength, normalized in-strength,
out-degree, and in-degree while destroying route identity.

The signed anatomical operator is column-normalized by absolute outgoing weight,
P[post, pre] = W[post, pre] / sum_post |W[post, pre]|.  We study the stable
discrete propagator A_s = beta I + gamma G_s P.  Because ||G_s P||_1 <= g_max,
choosing beta + gamma*g_max < 1 gives both states the same rigorous stability
bound without fitting task responses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, eigs


HERE = Path(__file__).resolve().parent
DEFAULT_GRAPH = HERE.parent / "flywire_spontaneous" / "Connectivity_783.parquet"
DEFAULT_NODES = HERE.parent / "flywire_spontaneous" / "Completeness_783_all_139255.csv"
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)

SOURCE_REPOSITORY = "https://github.com/philshiu/Drosophila_brain_model"
SOURCE_COMMIT = "91bdd1e7dcf193f3e7ca5a8933497fcef63b7960"
SOURCE_DOI = "10.1038/s41586-024-07763-9"
CONNECTOME_DOI = "10.1038/s41586-024-07558-y"
SOURCE_VERSION = 783
PUBLISHED_NEURON_COUNT = 139_255
MODEL_CONNECTED_NODE_COUNT = 138_639
ROOT_ID_MAPPINGS = {
    # v630 root: (v783 root, stable supervoxel used for the mapping)
    720575940620900446: (720575940639259967, 78818697598428726),
    720575940645521262: (720575940618238523, 78536329000064308),
    720575940626307902: (720575940625054647, 78536191829712049),
}


# Shiu et al., figures.ipynb, Fig. 1D: right-hemisphere labellar sugar GRNs.
SUGAR_GRNS = [
    720575940624963786, 720575940630233916, 720575940637568838,
    720575940638202345, 720575940617000768, 720575940630797113,
    720575940632889389, 720575940621754367, 720575940621502051,
    720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940639259967, 720575940617937543,
    720575940632425919, 720575940633143833, 720575940612670570,
    720575940628853239, 720575940629176663, 720575940611875570,
]

# Shiu et al., figures.ipynb, Fig. 5B: Johnston-organ CE neurons.
JON_CE = [
    720575940619341105, 720575940630122015, 720575940611061526,
    720575940615848788, 720575940628444667, 720575940627941431,
    720575940632449619, 720575940650244342, 720575940631866508,
    720575940638681845, 720575940628978450, 720575940609522461,
    720575940621442224, 720575940602506208, 720575940629022149,
    720575940627109991, 720575940630020111, 720575940615986459,
    720575940618684481, 720575940620382889, 720575940630080071,
    720575940626565455, 720575940630319671, 720575940602720940,
    720575940630564179, 720575940637632419, 720575940615809349,
    720575940626042149, 720575940637054835, 720575940602132509,
    720575940614188149, 720575940616951124, 720575940628101126,
    720575940629055721, 720575940616589878, 720575940622449388,
    720575940614427195, 720575940625797617, 720575940638664437,
    720575940618467195, 720575940621729757, 720575940613971485,
    720575940627585688, 720575940629650997, 720575940630059847,
    720575940608742409, 720575940614351477, 720575940633153375,
    720575940622937528, 720575940604753437, 720575940611783464,
    720575940618599872, 720575940609541917, 720575940637410869,
    720575940630070343, 720575940621397417, 720575940614035485,
    720575940610018266, 720575940625054647, 720575940634634606,
    720575940614060829, 720575940624799290, 720575940641921421,
    720575940623298559, 720575940625559358, 720575940629138959,
    720575940621625597, 720575940625962568, 720575940632767383,
    720575940624915230,
]

# Shiu et al., figures.ipynb, Fig. 5B: additional Johnston-organ subtypes.
JON_F = [
    720575940606239243, 720575940626956777, 720575940604973746,
    720575940622222856, 720575940642517284, 720575940629719404,
    720575940616613022, 720575940604299454, 720575940615473186,
    720575940622217992, 720575940606800341, 720575940629267498,
    720575940637366335, 720575940624224408, 720575940609543197,
    720575940633364179, 720575940629502009, 720575940606431189,
    720575940625733960, 720575940638529525, 720575940617524053,
    720575940628935564, 720575940624308355, 720575940631170346,
    720575940627704375, 720575940625885512, 720575940614929245,
    720575940647493241, 720575940618888368, 720575940625087546,
    720575940606657493, 720575940617273560, 720575940640591861,
    720575940639410035, 720575940621532413, 720575940627523584,
    720575940621521917, 720575940621097398, 720575940625915338,
    720575940606222428, 720575940627868471, 720575940622179497,
    720575940608297774, 720575940614026269, 720575940613012959,
    720575940628100614, 720575940606611401, 720575940628649465,
    720575940610008217, 720575940623791152, 720575940625571240,
    720575940634923621, 720575940609530653, 720575940635968745,
    720575940625703434, 720575940613105311, 720575940629386819,
    720575940623077389, 720575940625763015, 720575940628359017,
]

JON_D_M = [
    720575940630834171, 720575940622892988, 720575940621289537,
    720575940641395163, 720575940616064546, 720575940628978409,
    720575940652566177, 720575940627493096, 720575940619085397,
    720575940635545310, 720575940645728803, 720575940629141775,
    720575940626557995, 720575940631098338, 720575940639904475,
    720575940635067034,
]

JON_ALL = JON_CE + JON_F + JON_D_M

MN9 = [720575940660219265, 720575940618238523]
GROOMING_DNS = [720575940616185531, 720575940629806974]


@dataclass(frozen=True)
class Task:
    name: str
    input_ids: tuple[int, ...]
    output_ids: tuple[int, ...]


TASKS = (
    Task("feeding", tuple(SUGAR_GRNS), tuple(MN9)),
    Task("grooming", tuple(JON_ALL), tuple(GROOMING_DNS)),
)


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_full_graph(graph_path: Path, node_path: Path):
    """Load the published v783 neuron set and full model-ready edge graph."""
    node_frame = pd.read_csv(node_path, index_col=0)
    node_ids = node_frame.index.to_numpy(np.int64)
    n = len(node_ids)
    id_to_index = {int(node_id): index for index, node_id in enumerate(node_ids)}

    table = pq.read_table(
        graph_path,
        columns=[
            "Presynaptic_ID",
            "Postsynaptic_ID",
            "Presynaptic_Index",
            "Postsynaptic_Index",
            "Excitatory x Connectivity",
        ],
    )
    pre_ids = table.column("Presynaptic_ID").to_numpy(zero_copy_only=False)
    post_ids = table.column("Postsynaptic_ID").to_numpy(zero_copy_only=False)
    pre = table.column("Presynaptic_Index").to_numpy(zero_copy_only=False).astype(np.int32)
    post = table.column("Postsynaptic_Index").to_numpy(zero_copy_only=False).astype(np.int32)
    weight = table.column("Excitatory x Connectivity").to_numpy(
        zero_copy_only=False
    ).astype(np.float32)

    if int(max(pre.max(), post.max())) >= n:
        raise AssertionError("edge index exceeds completeness-table length")
    check = np.linspace(0, len(pre) - 1, 256, dtype=np.int64)
    if not np.array_equal(node_ids[pre[check]], pre_ids[check]):
        raise AssertionError("presynaptic ID/index mapping is inconsistent")
    if not np.array_equal(node_ids[post[check]], post_ids[check]):
        raise AssertionError("postsynaptic ID/index mapping is inconsistent")

    out_abs = np.bincount(pre, weights=np.abs(weight), minlength=n).astype(np.float32)
    nonzero = out_abs[pre] > 0
    pre = pre[nonzero]
    post = post[nonzero]
    weight = weight[nonzero]
    normalized = weight / out_abs[pre]
    p_signed = sparse.csr_matrix(
        (normalized, (post, pre)), shape=(n, n), dtype=np.float32
    )
    p_signed.sum_duplicates()
    p_abs = abs(p_signed).tocsr()

    in_abs = np.asarray(p_abs.sum(axis=1)).ravel().astype(np.float32)
    out_degree = np.bincount(pre, minlength=n).astype(np.int32)
    in_degree = np.bincount(post, minlength=n).astype(np.int32)
    sign_class = np.zeros(n, dtype=np.int8)
    signed_out = np.bincount(pre, weights=weight, minlength=n)
    sign_class[signed_out > 0] = 1
    sign_class[signed_out < 0] = -1

    return {
        "n": n,
        "m": int(p_signed.nnz),
        "node_ids": node_ids,
        "id_to_index": id_to_index,
        "p_signed": p_signed,
        "p_abs": p_abs,
        "in_abs": in_abs,
        "out_abs": out_abs,
        "in_degree": in_degree,
        "out_degree": out_degree,
        "sign_class": sign_class,
    }


def unit_set_vector(indices: np.ndarray, n: int) -> np.ndarray:
    vector = np.zeros(n, dtype=np.float32)
    vector[indices] = 1.0 / math.sqrt(len(indices))
    return vector


def task_vectors(task: Task, graph):
    try:
        inputs = np.array([graph["id_to_index"][x] for x in task.input_ids], dtype=np.int32)
        outputs = np.array([graph["id_to_index"][x] for x in task.output_ids], dtype=np.int32)
    except KeyError as error:
        raise AssertionError(f"published task neuron missing from graph: {error}")
    return inputs, outputs, unit_set_vector(inputs, graph["n"]), unit_set_vector(outputs, graph["n"])


def cumulative_reach(matrix, vector, hops: int, damping: float) -> np.ndarray:
    current = vector.astype(np.float32, copy=True)
    total = np.zeros_like(current)
    coefficient = 1.0
    for _ in range(hops):
        current = matrix @ current
        total += coefficient * current
        coefficient *= damping
    return total


def corridor_score(graph, task: Task, hops: int, damping: float):
    inputs, outputs, b, c = task_vectors(task, graph)
    forward = cumulative_reach(graph["p_abs"], b, hops, damping)
    backward = cumulative_reach(graph["p_abs"].T.tocsr(), c, hops, damping)
    score = np.sqrt(np.maximum(forward, 0) * np.maximum(backward, 0))
    score[inputs] = 0
    score[outputs] = 0
    return score, inputs, outputs


def choose_route_specific_modules(
    feeding_score: np.ndarray,
    grooming_score: np.ndarray,
    module_size: int,
    ranking_rule: str = "contrast_sqrt",
    epsilon: float = 1e-20,
):
    total = feeding_score + grooming_score
    if ranking_rule == "contrast_sqrt":
        signed_priority = (feeding_score - grooming_score) / (total + epsilon)
        signed_priority *= np.sqrt(total)
    elif ranking_rule == "difference":
        signed_priority = feeding_score - grooming_score
    elif ranking_rule == "log_ratio_sqrt":
        signed_priority = np.log(
            (feeding_score + epsilon) / (grooming_score + epsilon)
        ) * np.sqrt(total)
    else:
        raise ValueError(f"unknown ranking rule: {ranking_rule}")
    feed_priority = np.where(signed_priority > 0, signed_priority, -np.inf)
    groom_priority = np.where(signed_priority < 0, -signed_priority, -np.inf)

    feed = np.argpartition(feed_priority, -module_size)[-module_size:]
    groom = np.argpartition(groom_priority, -module_size)[-module_size:]
    feed = feed[np.argsort(feed_priority[feed])[::-1]].astype(np.int32)
    groom = groom[np.argsort(groom_priority[groom])[::-1]].astype(np.int32)
    if len(np.intersect1d(feed, groom)):
        raise AssertionError("route-specific gain modules overlap")
    if not np.isfinite(feed_priority[feed[-1]]) or not np.isfinite(groom_priority[groom[-1]]):
        raise AssertionError("too few route-specific corridor neurons")
    return feed, groom


class GainPropagator:
    def __init__(self, p_signed, gain: np.ndarray, beta: float, gamma: float):
        self.p = p_signed
        self.gain = gain.astype(np.float32, copy=False)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.shape = p_signed.shape
        self.dtype = np.dtype(np.float32)

    def matvec(self, x: np.ndarray) -> np.ndarray:
        return self.beta * x + self.gamma * self.gain * (self.p @ x)

    def rmatvec(self, x: np.ndarray) -> np.ndarray:
        return self.beta * x + self.gamma * (self.p.T @ (self.gain * x))

    def matmat(self, x: np.ndarray) -> np.ndarray:
        return self.beta * x + self.gamma * self.gain[:, None] * (self.p @ x)

    def as_linear_operator(self) -> LinearOperator:
        return LinearOperator(
            self.shape,
            matvec=self.matvec,
            rmatvec=self.rmatvec,
            matmat=self.matmat,
            dtype=self.dtype,
        )


def make_gain(n: int, module: np.ndarray, gain_high: float) -> np.ndarray:
    gain = np.ones(n, dtype=np.float32)
    gain[module] = gain_high
    return gain


def route_response(operator: GainPropagator, b: np.ndarray, outputs: np.ndarray, steps: int):
    x = b.astype(np.float32, copy=True)
    values = []
    for step in range(steps + 1):
        values.append(float(np.dot(x[outputs], x[outputs])))
        if step < steps:
            x = operator.matvec(x)
    values = np.asarray(values)
    peak = int(np.argmax(values))
    return {"peak": float(values[peak]), "peak_step": peak, "curve": values}


def finite_horizon_cost(
    operator: GainPropagator,
    probes: np.ndarray,
    steps: int,
):
    x = probes.astype(np.float32, copy=True)
    curve = [float(np.mean(np.sum(x * x, axis=0)) / operator.shape[0])]
    for _ in range(steps):
        x = operator.matmat(x)
        curve.append(float(np.mean(np.sum(x * x, axis=0)) / operator.shape[0]))
    return float(np.sum(curve)), np.asarray(curve)


def spectral_summary(operator: GainPropagator):
    linear = operator.as_linear_operator()
    values = eigs(linear, k=4, which="LM", return_eigenvectors=False, tol=2e-4, maxiter=800)
    radius = float(np.max(np.abs(values)))
    return radius, [[float(z.real), float(z.imag)] for z in values]


def stratification_labels(graph, candidates: np.ndarray, bins: int = 2):
    """Joint covariate strata without mixing quantities on different scales."""
    def quantile_bin(values):
        edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
        if len(edges) <= 2:
            return np.zeros(len(values), dtype=np.int16)
        return np.digitize(values, edges[1:-1], right=True).astype(np.int16)

    features = (
        np.log1p(graph["out_abs"][candidates]),
        np.log1p(graph["in_abs"][candidates]),
        np.log1p(graph["out_degree"][candidates]),
        np.log1p(graph["in_degree"][candidates]),
    )
    labels = graph["sign_class"][candidates].astype(np.int32) + 1
    for values in features:
        labels = labels * bins + quantile_bin(values)
    return labels


def stratified_permutation(
    feed: np.ndarray,
    groom: np.ndarray,
    graph,
    rng: np.random.Generator,
    bins: int = 2,
):
    union = np.concatenate([feed, groom])
    labels = np.concatenate([
        np.ones(len(feed), dtype=bool),
        np.zeros(len(groom), dtype=bool),
    ])
    strata = stratification_labels(graph, union, bins=bins)
    permuted = labels.copy()
    for stratum in np.unique(strata):
        positions = np.flatnonzero(strata == stratum)
        permuted[positions] = rng.permutation(labels[positions])
    if int(permuted.sum()) != len(feed):
        raise AssertionError("stratified permutation changed module size")
    return union[permuted].astype(np.int32), union[~permuted].astype(np.int32)


def switch_metrics(response):
    eps = np.finfo(float).tiny
    feed_ratio = (response["feeding"]["state_feed"]["peak"] + eps) / (
        response["feeding"]["state_groom"]["peak"] + eps
    )
    groom_ratio = (response["grooming"]["state_groom"]["peak"] + eps) / (
        response["grooming"]["state_feed"]["peak"] + eps
    )
    log_switch = 0.5 * (math.log(feed_ratio) + math.log(groom_ratio))
    return {
        "feeding_selectivity": float(feed_ratio),
        "grooming_selectivity": float(groom_ratio),
        "geometric_switch_factor": float(math.exp(log_switch)),
        "log_switch": float(log_switch),
    }


def evaluate_pair(graph, feed_module, groom_module, parameters, task_cache):
    gain_feed = make_gain(graph["n"], feed_module, parameters["gain_high"])
    gain_groom = make_gain(graph["n"], groom_module, parameters["gain_high"])
    op_feed = GainPropagator(graph["p_signed"], gain_feed, parameters["beta"], parameters["gamma"])
    op_groom = GainPropagator(graph["p_signed"], gain_groom, parameters["beta"], parameters["gamma"])
    response = {}
    for name, cached in task_cache.items():
        response[name] = {
            "state_feed": route_response(op_feed, cached["b"], cached["outputs"], parameters["response_steps"]),
            "state_groom": route_response(op_groom, cached["b"], cached["outputs"], parameters["response_steps"]),
        }
    return response, op_feed, op_groom


def serializable_response(response):
    result = {}
    for task, states in response.items():
        result[task] = {}
        for state, values in states.items():
            result[task][state] = {
                "peak": values["peak"],
                "peak_step": values["peak_step"],
                "curve": values["curve"].tolist(),
            }
    return result


def plot_results(summary: dict, output: Path):
    red = "#C62828"
    blue = "#1565C0"
    grey = "#90A4AE"
    plt.rcParams.update({
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "axes.linewidth": 0.9,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })
    fig, axes = plt.subplots(1, 4, figsize=(7.15, 1.95), constrained_layout=True)

    ax = axes[0]
    values = [summary["graph"]["neurons"], summary["graph"]["edges"]]
    ax.bar([0, 1], values, bottom=1, color=[grey, "#546E7A"])
    ax.set_yscale("log")
    ax.set_ylim(8e4, 3.5e7)
    ax.set_xticks([0, 1], ["neurons", "edges"])
    ax.set_ylabel("full-release count")
    ax.set_title("(a) Full FlyWire v783 graph")
    labels = [f"{values[0]:,}", f"{values[1] / 1e6:.2f} M"]
    for x, value, label in zip([0, 1], values, labels):
        ax.text(x, value * 1.10, label, ha="center", va="bottom", fontsize=6.4)

    ax = axes[1]
    global_values = summary["global_matching"]
    x = np.arange(2)
    width = 0.34
    ax.bar(x - width / 2, [global_values["spectral_radius_feed"], global_values["cost_feed"]],
           width, color=red, label="feed state")
    ax.bar(x + width / 2, [global_values["spectral_radius_groom"], global_values["cost_groom"]],
           width, color=blue, label="groom state")
    ax.set_xticks(x, [r"$\rho(A)$", r"$C_H/C_H^{\rm feed}$"])
    ax.set_ylim(0.75, 1.055)
    ax.set_title("(b) Global summaries match")
    ax.legend(frameon=False, fontsize=6.0, loc="upper left")

    def latex_sci(value: float) -> str:
        exponent = int(math.floor(math.log10(abs(value))))
        coefficient = value / 10**exponent
        return rf"{coefficient:.1f}\times10^{{{exponent}}}"

    radius_difference = global_values["spectral_radius_relative_difference"]
    # ARPACK is run at tol=2e-4. Do not display a sub-tolerance raw difference
    # as though it were a seven-digit estimate; the common induced-norm bound is
    # the rigorous stability guarantee used by the analysis.
    radius_difference_label = (
        r"<10^{-4}" if radius_difference < 1e-4 else f"={latex_sci(radius_difference)}"
    )

    ax.text(
        0.98,
        0.04,
        rf"$\Delta\rho/\rho{radius_difference_label}$"
        "\n"
        rf"$\Delta C_H/C_H={latex_sci(global_values['cost_relative_difference'])}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.9,
    )

    ax = axes[2]
    transfer = np.array([
        [summary["response"]["feeding"]["state_feed"]["peak"],
         summary["response"]["feeding"]["state_groom"]["peak"]],
        [summary["response"]["grooming"]["state_feed"]["peak"],
         summary["response"]["grooming"]["state_groom"]["peak"]],
    ])
    normalized = transfer / np.maximum(transfer.max(axis=1, keepdims=True), 1e-30)
    ax.imshow(normalized, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks([0, 1], ["feed", "groom"])
    ax.set_yticks([0, 1], ["feeding", "grooming"])
    ax.set_xlabel("gain state")
    ax.set_ylabel("task route")
    ax.set_title("(c) Route transfer")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{normalized[i, j]:.2f}", ha="center", va="center",
                    color="white" if normalized[i, j] > 0.55 else "0.2")

    ax = axes[3]
    null = np.asarray(summary["null"]["switch_factors"])
    observed = summary["switch"]["geometric_switch_factor"]
    ax.hist(null, bins=min(20, max(8, len(null) // 4)), color=grey, edgecolor="white")
    ax.axvline(observed, color=red, lw=2, label=f"observed {observed:.2f}×")
    ax.set_xlim(0, max(observed, float(null.max())) * 1.10)
    ax.set_xlabel("routing switch factor")
    ax.set_ylabel("stratified nulls")
    ax.set_title("(d) Route labels matter")
    ax.legend(frameon=False, fontsize=6.0, loc="upper right")
    ax.text(0.98, 0.04, f"$p={summary['null']['p_one_sided']:.3g}$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.8)

    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")


def run(args):
    started = time.time()
    graph = load_full_graph(args.graph, args.nodes)
    task_cache = {}
    corridor = {}
    excluded = set()
    for task in TASKS:
        score, inputs, outputs = corridor_score(graph, task, args.corridor_hops, args.damping)
        task_cache[task.name] = {
            "inputs": inputs,
            "outputs": outputs,
            "b": unit_set_vector(inputs, graph["n"]),
        }
        corridor[task.name] = score
        excluded.update(inputs.tolist())
        excluded.update(outputs.tolist())
    for index in excluded:
        corridor["feeding"][index] = 0
        corridor["grooming"][index] = 0

    feed_module, groom_module = choose_route_specific_modules(
        corridor["feeding"], corridor["grooming"], args.module_size,
        ranking_rule=args.ranking_rule, epsilon=args.ranking_epsilon,
    )
    parameters = {
        "gain_high": args.gain_high,
        "beta": args.beta,
        "gamma": (args.stability_bound - args.beta) / args.gain_high,
        "stability_bound": args.stability_bound,
        "response_steps": args.response_steps,
    }
    if parameters["gamma"] <= 0:
        raise ValueError("stability_bound must exceed beta")

    response, op_feed, op_groom = evaluate_pair(
        graph, feed_module, groom_module, parameters, task_cache
    )
    switch = switch_metrics(response)

    rng = np.random.default_rng(args.seed)
    null_switch = []
    for _ in range(args.nulls):
        null_feed, null_groom = stratified_permutation(
            feed_module,
            groom_module,
            graph,
            rng,
            bins=args.null_feature_bins,
        )
        null_response, _, _ = evaluate_pair(
            graph, null_feed, null_groom, parameters, task_cache
        )
        null_switch.append(switch_metrics(null_response)["geometric_switch_factor"])
    null_switch = np.asarray(null_switch)
    p_one_sided = float((1 + np.sum(null_switch >= switch["geometric_switch_factor"])) / (args.nulls + 1))

    probes = rng.choice(
        np.array([-1.0, 1.0], dtype=np.float32),
        size=(graph["n"], args.cost_probes),
    )
    raw_cost_feed, cost_curve_feed = finite_horizon_cost(op_feed, probes, args.cost_steps)
    raw_cost_groom, cost_curve_groom = finite_horizon_cost(op_groom, probes, args.cost_steps)
    radius_feed, eig_feed = spectral_summary(op_feed)
    radius_groom, eig_groom = spectral_summary(op_groom)
    cost_reference = raw_cost_feed

    module_union = np.concatenate([feed_module, groom_module])
    summary = {
        "provenance": {
            "source_repository": SOURCE_REPOSITORY,
            "source_commit": SOURCE_COMMIT,
            "source_doi": SOURCE_DOI,
            "connectome_doi": CONNECTOME_DOI,
            "materialization_version": SOURCE_VERSION,
            "graph_path": str(args.graph),
            "node_path": str(args.nodes),
            "graph_sha256": sha256(args.graph),
            "node_sha256": sha256(args.nodes),
            "root_id_mappings": {
                str(old): {
                    "v783_root": int(new),
                    "stable_supervoxel": int(supervoxel),
                }
                for old, (new, supervoxel) in ROOT_ID_MAPPINGS.items()
            },
            "interpretation": (
                "Connectome-constrained in-silico gain intervention; not an in-vivo "
                "measurement of state-dependent neuronal gain."
            ),
        },
        "graph": {
            "neurons": graph["n"],
            "edges": graph["m"],
            "connected_model_nodes": MODEL_CONNECTED_NODE_COUNT,
            "isolated_published_neurons_appended": (
                PUBLISHED_NEURON_COUNT - MODEL_CONNECTED_NODE_COUNT
            ),
            "uses_all_edges": True,
        },
        "tasks": {
            name: {
                "input_count": int(len(cached["inputs"])),
                "output_count": int(len(cached["outputs"])),
                "input_ids": [int(graph["node_ids"][i]) for i in cached["inputs"]],
                "output_ids": [int(graph["node_ids"][i]) for i in cached["outputs"]],
            }
            for name, cached in task_cache.items()
        },
        "parameters": {
            **parameters,
            "module_size_each": args.module_size,
            "ranking_rule": args.ranking_rule,
            "ranking_epsilon": args.ranking_epsilon,
            "corridor_hops": args.corridor_hops,
            "corridor_damping": args.damping,
            "nulls": args.nulls,
            "null_seed": args.seed,
            "null_feature_bins": args.null_feature_bins,
            "cost_probes": args.cost_probes,
            "cost_steps": args.cost_steps,
        },
        "gain_modules": {
            "feeding_ids": [int(graph["node_ids"][i]) for i in feed_module],
            "grooming_ids": [int(graph["node_ids"][i]) for i in groom_module],
            "overlap": int(len(np.intersect1d(feed_module, groom_module))),
            "same_size": bool(len(feed_module) == len(groom_module)),
            "same_gain_histogram": True,
            "union_sign_counts": {
                str(sign): int(np.sum(graph["sign_class"][module_union] == sign))
                for sign in (-1, 0, 1)
            },
        },
        "response": serializable_response(response),
        "switch": switch,
        "global_matching": {
            "common_induced_one_norm_bound": args.stability_bound,
            "spectral_radius_feed": radius_feed,
            "spectral_radius_groom": radius_groom,
            "spectral_radius_relative_difference": abs(radius_feed - radius_groom) / max(radius_feed, radius_groom),
            "leading_eigenvalues_feed": eig_feed,
            "leading_eigenvalues_groom": eig_groom,
            "raw_finite_horizon_cost_feed": raw_cost_feed,
            "raw_finite_horizon_cost_groom": raw_cost_groom,
            "cost_feed": raw_cost_feed / cost_reference,
            "cost_groom": raw_cost_groom / cost_reference,
            "cost_relative_difference": abs(raw_cost_feed - raw_cost_groom) / max(raw_cost_feed, raw_cost_groom),
            "cost_curve_feed": cost_curve_feed.tolist(),
            "cost_curve_groom": cost_curve_groom.tolist(),
        },
        "null": {
            "description": (
                "Gain-module labels shuffled within neurotransmitter-sign and "
                "joint strata formed by separately quantiling raw out-strength, "
                "normalized in-strength, out-degree, and in-degree."
            ),
            "switch_factors": null_switch.tolist(),
            "median": float(np.median(null_switch)),
            "q95": float(np.quantile(null_switch, 0.95)),
            "p_one_sided": p_one_sided,
        },
        "predeclared_gates": {
            "observed_exceeds_95pct_null": bool(switch["geometric_switch_factor"] > np.quantile(null_switch, 0.95)),
            "both_routes_prefer_assigned_state": bool(
                switch["feeding_selectivity"] > 1.0
                and switch["grooming_selectivity"] > 1.0
            ),
            "permutation_p_at_most_0p05": bool(p_one_sided <= 0.05),
            "spectral_radius_difference_below_2pct": bool(abs(radius_feed - radius_groom) / max(radius_feed, radius_groom) < 0.02),
            "cost_difference_below_5pct": bool(abs(raw_cost_feed - raw_cost_groom) / max(raw_cost_feed, raw_cost_groom) < 0.05),
        },
        "runtime_seconds": float(time.time() - started),
    }
    summary["predeclared_gates"]["all_pass"] = bool(all(summary["predeclared_gates"].values()))

    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    plot_results(summary, args.figure)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--nodes", type=Path, default=DEFAULT_NODES)
    parser.add_argument("--module-size", type=int, default=256)
    parser.add_argument(
        "--ranking-rule",
        choices=("contrast_sqrt", "difference", "log_ratio_sqrt"),
        default="contrast_sqrt",
    )
    parser.add_argument("--ranking-epsilon", type=float, default=1e-20)
    parser.add_argument("--gain-high", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=0.70)
    parser.add_argument("--stability-bound", type=float, default=0.98)
    parser.add_argument("--corridor-hops", type=int, default=8)
    parser.add_argument("--damping", type=float, default=0.85)
    parser.add_argument("--response-steps", type=int, default=24)
    parser.add_argument("--cost-steps", type=int, default=40)
    parser.add_argument("--cost-probes", type=int, default=16)
    parser.add_argument("--nulls", type=int, default=999)
    parser.add_argument(
        "--null-feature-bins",
        type=int,
        default=2,
        help="Quantile bins per covariate in the joint four-feature null strata.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=HERE / "flywire_gain_routing_results.json")
    parser.add_argument("--figure", type=Path, default=FIG_DIR / "fig_flywire_gain_routing.pdf")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = run(args)
    print(json.dumps({
        "graph": summary["graph"],
        "switch": summary["switch"],
        "global_matching": {
            key: summary["global_matching"][key]
            for key in (
                "spectral_radius_feed",
                "spectral_radius_groom",
                "spectral_radius_relative_difference",
                "cost_feed",
                "cost_groom",
                "cost_relative_difference",
            )
        },
        "null": {
            "median": summary["null"]["median"],
            "q95": summary["null"]["q95"],
            "p_one_sided": summary["null"]["p_one_sided"],
        },
        "gates": summary["predeclared_gates"],
        "runtime_seconds": summary["runtime_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
