# -*- coding: utf-8 -*-
"""
ДЗ 13. Симуляция распространения болезни (SI) на сети аэропортов.

Блоки:
  0) Загрузка данных
  1) Функция SI-симуляции
  2) Влияние вероятности p (графики)
  3) Граф NetworkX + медианное время + Spearman

Запуск:
  python hw13_disease_spread.py
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

# =============================================================================
# Блок 0. Настройки и пути
# =============================================================================
BASE = Path(__file__).resolve().parent
RANDOM_SEED = 42
START_NODE = 0  # Allentown
P_VALUES = [0.01, 0.05, 0.1, 0.5, 1.0]
N_RUNS_P = 10
N_RUNS_MEDIAN = 50
P_MEDIAN = 0.5
TWELVE_HOURS = 12 * 3600


def load_data():
    """Читает аэропорты и рейсы; сортирует рейсы по времени вылета."""
    airport_data = pd.read_csv(BASE / "airport_data.csv")
    flight_data = pd.read_csv(BASE / "flight.txt", sep=" ")
    flight_data = flight_data.sort_values(
        ["StartTime", "EndTime"], kind="mergesort"
    ).reset_index(drop=True)
    id_to_name = airport_data.set_index("id")["airport name"].to_dict()
    return airport_data, flight_data, id_to_name


# =============================================================================
# Блок 1. SI-симуляция одного прохода
# =============================================================================
def simulate_infection(flights_df, start_node=0, p=0.01, airport_names=None, seed=None):
    """
    Проход по всем рейсам (itertuples).
    Если Source заражён, а Destination ещё нет — с вероятностью p заражаем
    destination в момент EndTime.

    Возвращает: {время_заражения: название_аэропорта}
    Дополнительно пишет last_infection_times: {id: время}.
    """
    if airport_names is None:
        raise ValueError("Нужен словарь airport_names (id -> название)")
    if seed is not None:
        random.seed(seed)

    infected = {start_node}
    start_flights = flights_df.loc[flights_df["Source"] == start_node, "StartTime"]
    start_time = (
        int(start_flights.min())
        if len(start_flights)
        else int(flights_df["StartTime"].min())
    )

    infection_log = {start_time: airport_names[start_node]}
    infection_times = {start_node: start_time}

    for row in flights_df.itertuples(index=False):
        src, dst = row.Source, row.Destination
        if src in infected and dst not in infected and random.random() < p:
            infected.add(dst)
            t = int(row.EndTime)
            infection_times[dst] = t
            infection_log[t] = airport_names[dst]

    simulate_infection.last_infection_times = infection_times
    return infection_log


def infection_times_by_airport(flights_df, airport_names, start_node=0, p=0.01, seed=None):
    """Обёртка: id аэропорта -> время заражения."""
    simulate_infection(
        flights_df, start_node=start_node, p=p, airport_names=airport_names, seed=seed
    )
    return dict(simulate_infection.last_infection_times)


def infected_percent_curve(infection_times, time_grid, n_airports):
    """% заражённых аэропортов к каждому моменту time_grid."""
    times = np.array(sorted(infection_times.values()), dtype=np.int64)
    counts = np.searchsorted(times, time_grid, side="right")
    return 100.0 * counts / n_airports


# =============================================================================
# Блок 2–3. Граф NetworkX
# =============================================================================
def build_graph(flight_data, airport_ids):
    """
    Ненаправленный граф: вес ребра =
    (рейсы A↔B) / (всего рейсов в датасете).
    """
    total_flights = len(flight_data)
    pair_counts = defaultdict(int)
    for row in flight_data.itertuples(index=False):
        a, b = int(row.Source), int(row.Destination)
        if a == b:
            continue
        pair_counts[(min(a, b), max(a, b))] += 1

    G = nx.Graph()
    G.add_nodes_from(airport_ids)
    for (a, b), cnt in pair_counts.items():
        G.add_edge(a, b, weight=cnt / total_flights, n_flights=cnt)
    return G


# =============================================================================
# main: части 2 и 3 + сохранение графиков
# =============================================================================
def main():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.figsize"] = (10, 5)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    airport_data, flight_data, id_to_name = load_data()
    n_airports = airport_data.shape[0]
    t_min = int(flight_data["StartTime"].min())
    t_max = int(flight_data["EndTime"].max())
    time_grid = np.arange(t_min, t_max + TWELVE_HOURS, TWELVE_HOURS)

    print(f"Аэропортов: {n_airports}, рейсов: {len(flight_data)}")
    print(f"Старт: {airport_data.loc[0, 'city']} / {id_to_name[START_NODE]}")

    # --- Часть 1: демо ---
    demo = simulate_infection(
        flight_data, start_node=START_NODE, p=0.5, airport_names=id_to_name, seed=0
    )
    print(f"Демо (p=0.5): заражено {len(demo)} аэропортов")

    # --- Часть 2: влияние p ---
    results_by_p = {}
    for p in P_VALUES:
        curves, finals = [], []
        for run in range(N_RUNS_P):
            times = infection_times_by_airport(
                flight_data,
                id_to_name,
                start_node=START_NODE,
                p=p,
                seed=RANDOM_SEED + run,
            )
            curves.append(infected_percent_curve(times, time_grid, n_airports))
            finals.append(len(times))
        results_by_p[p] = {
            "mean_curve": np.mean(curves, axis=0),
            "std_curve": np.std(curves, axis=0),
            "mean_final": float(np.mean(finals)),
        }
        print(
            f"p={p:>4}: средний финал {results_by_p[p]['mean_final']:.1f}/{n_airports} "
            f"({100 * results_by_p[p]['mean_final'] / n_airports:.1f}%)"
        )

    hours_from_start = (time_grid - t_min) / 3600.0
    fig, ax = plt.subplots(figsize=(11, 6))
    for p in P_VALUES:
        mean_c = results_by_p[p]["mean_curve"]
        std_c = results_by_p[p]["std_curve"]
        ax.plot(hours_from_start, mean_c, label=f"p={p}")
        ax.fill_between(hours_from_start, mean_c - std_c, mean_c + std_c, alpha=0.15)
    ax.set_xlabel("Часы от начала симуляции")
    ax.set_ylabel("Средний % заражённых аэропортов")
    ax.set_title("Влияние вероятности p на скорость распространения (10 запусков)")
    ax.legend(title="p")
    ax.set_ylim(0, 105)
    plt.tight_layout()
    fig.savefig(BASE / "part2_infection_vs_p.png", dpi=140)
    plt.close(fig)

    # --- Часть 3: граф + медиана + Spearman ---
    G = build_graph(flight_data, airport_data["id"].tolist())
    print(f"Граф: {G.number_of_nodes()} вершин, {G.number_of_edges()} рёбер")

    infection_runs = defaultdict(list)
    for run in range(N_RUNS_MEDIAN):
        times = infection_times_by_airport(
            flight_data,
            id_to_name,
            start_node=START_NODE,
            p=P_MEDIAN,
            seed=RANDOM_SEED + 1000 + run,
        )
        for node, t in times.items():
            infection_runs[node].append(t)

    median_hours = {
        node: (float(np.median(ts)) - t_min) / 3600.0
        for node, ts in infection_runs.items()
    }
    print(f"Заражались хотя бы раз: {len(median_hours)} / {n_airports}")

    clustering = nx.clustering(G)
    degree = dict(G.degree())
    betweenness = nx.betweenness_centrality(G)

    metrics_df = airport_data[["id", "city", "airport name", "symbol"]].copy()
    metrics_df["median_infection_hours"] = metrics_df["id"].map(median_hours)
    metrics_df["clustering"] = metrics_df["id"].map(clustering)
    metrics_df["degree"] = metrics_df["id"].map(degree)
    metrics_df["betweenness"] = metrics_df["id"].map(betweenness)
    metrics_df.to_csv(BASE / "part3_metrics.csv", index=False)

    plot_df = metrics_df.dropna(subset=["median_infection_hours"]).copy()
    metric_cols = ["clustering", "degree", "betweenness"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    spearman_results = {}
    for ax, col in zip(axes, metric_cols):
        x = plot_df["median_infection_hours"]
        y = plot_df[col]
        ax.scatter(x, y, alpha=0.65, edgecolor="none", s=35)
        coef, pval = spearmanr(x, y)
        spearman_results[col] = (float(coef), float(pval))
        ax.set_xlabel("Медианное время заражения, часы")
        ax.set_ylabel(col)
        ax.set_title(f"{col}\nSpearman rho={coef:.3f}, p={pval:.2e}")
    plt.suptitle(
        "Метрики графа vs медианное время заражения (p=0.5, 50 симуляций)", y=1.02
    )
    plt.tight_layout()
    fig.savefig(BASE / "part3_scatter_metrics.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    ranked = sorted(
        spearman_results.items(), key=lambda kv: abs(kv[1][0]), reverse=True
    )
    print("Корреляции Спирмена с медианным временем заражения:")
    for col, (coef, pval) in ranked:
        print(f"  {col:14s}: rho={coef:+.4f}, p={pval:.3e}")
    print(f"Сильнее всего |rho|: {ranked[0][0]}")

    summary = {
        "part2_mean_final": {str(k): v["mean_final"] for k, v in results_by_p.items()},
        "spearman": {k: {"rho": v[0], "p": v[1]} for k, v in spearman_results.items()},
        "strongest": ranked[0][0],
        "n_infected_median_runs": len(median_hours),
        "n_airports": n_airports,
        "n_edges": G.number_of_edges(),
    }
    (BASE / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Готово. Графики и CSV сохранены в", BASE)


if __name__ == "__main__":
    main()
