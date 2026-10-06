"""
Генератор синтетического датасета производственных заданий (ASPMK-like).

Имитирует таблицу task_assignments + связанные production_tasks / route_sheets
из реального ERP-проекта: рабочий выполняет операцию над изделием в цехе,
у операции есть нормо-время (norm_minutes), и мы хотим предсказать фактическое
время выполнения (actual_minutes) по признакам задания.

Запуск:
    python generate_dataset.py
Результат:
    data/production_tasks.csv
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

N_ROWS = 12000

WORKSHOPS = ["Цех раскроя", "Цех сборки", "Цех упаковки", "Цех покраски", "Цех контроля"]
SPECIALITIES = ["Раскройщик", "Сборщик", "Упаковщик", "Маляр", "Контролёр ОТК"]
# у каждого цеха своя основная специальность (реалистичная связь)
WORKSHOP_SPECIALITY = dict(zip(WORKSHOPS, SPECIALITIES))

PRODUCT_GROUPS = ["Короб картонный", "Паллета деревянная", "Плёнка стрейч", "Ящик пластиковый", "Упаковка подарочная"]

N_WORKERS = 60
N_PRODUCTS = 40

# --- Рабочие: у каждого своя "истинная" скорость (скрытый навык), не видна модели напрямую ---
worker_ids = np.arange(1, N_WORKERS + 1)
worker_speciality = RNG.choice(SPECIALITIES, size=N_WORKERS)
# skill_factor < 1 = быстрее нормы, > 1 = медленнее нормы
worker_skill = RNG.normal(loc=1.0, scale=0.18, size=N_WORKERS).clip(0.6, 1.6)
worker_experience_years = RNG.integers(0, 20, size=N_WORKERS)
# опыт немного улучшает скорость (снижает skill_factor), но с насыщением
worker_skill = (worker_skill - 0.01 * np.minimum(worker_experience_years, 10)).clip(0.55, 1.6)

workers_df = pd.DataFrame({
    "worker_id": worker_ids,
    "speciality": worker_speciality,
    "experience_years": worker_experience_years,
    "_skill_factor": worker_skill,  # приставка _ = скрытая переменная, не для обучения "в лоб"
})

# --- Изделия: у каждого своя сложность и базовая норма времени на операцию ---
product_ids = np.arange(1, N_PRODUCTS + 1)
product_group = RNG.choice(PRODUCT_GROUPS, size=N_PRODUCTS)
product_complexity = RNG.uniform(0.7, 2.0, size=N_PRODUCTS)  # множитель сложности

products_df = pd.DataFrame({
    "product_id": product_ids,
    "product_group": product_group,
    "_complexity": product_complexity,
})

rows = []
start_date = pd.Timestamp("2025-01-06")  # понедельник

for i in range(N_ROWS):
    workshop = RNG.choice(WORKSHOPS)
    required_speciality = WORKSHOP_SPECIALITY[workshop]

    # рабочий выбирается из тех, у кого подходящая специальность (с редкими исключениями — подмена)
    matching = workers_df[workers_df["speciality"] == required_speciality]
    if len(matching) == 0 or RNG.random() < 0.05:
        worker = workers_df.sample(1, random_state=int(RNG.integers(0, 1_000_000))).iloc[0]
        speciality_mismatch = int(worker["speciality"] != required_speciality)
    else:
        worker = matching.sample(1, random_state=int(RNG.integers(0, 1_000_000))).iloc[0]
        speciality_mismatch = 0

    product = products_df.sample(1, random_state=int(RNG.integers(0, 1_000_000))).iloc[0]

    quantity = int(RNG.integers(5, 500))

    # норма времени на единицу зависит от сложности изделия и цеха
    base_minute_per_unit = RNG.uniform(0.8, 3.5) * product["_complexity"]
    norm_minutes = max(15, round(quantity * base_minute_per_unit))

    # день недели и смена — влияют на факт (пятница/понедельник обычно медленнее)
    day_offset = int(RNG.integers(0, 260))
    task_date = start_date + pd.Timedelta(days=day_offset)
    weekday = task_date.weekday()  # 0=Mon .. 6=Sun
    is_late_week = int(weekday in (0, 4))  # понедельник и пятница

    shift = RNG.choice(["day", "night"], p=[0.75, 0.25])
    is_night = int(shift == "night")

    # --- "истинная" формула фактического времени (то, что модель должна выучить) ---
    fatigue_penalty = 1.08 if is_late_week else 1.0
    night_penalty = 1.12 if is_night else 1.0
    mismatch_penalty = 1.35 if speciality_mismatch else 1.0

    # эффект перегрузки: если задание крупное (quantity выше медианы) — небольшая просадка эффективности
    overload_penalty = 1.0 + 0.05 * (quantity > 300)

    true_factor = (
        worker["_skill_factor"]
        * fatigue_penalty
        * night_penalty
        * mismatch_penalty
        * overload_penalty
    )

    noise = RNG.normal(loc=1.0, scale=0.12)
    actual_minutes = max(5, round(norm_minutes * true_factor * noise))

    rows.append({
        "task_id": i + 1,
        "task_date": task_date.date().isoformat(),
        "weekday": weekday,
        "shift": shift,
        "workshop": workshop,
        "worker_id": int(worker["worker_id"]),
        "worker_speciality": worker["speciality"],
        "required_speciality": required_speciality,
        "speciality_mismatch": speciality_mismatch,
        "worker_experience_years": int(worker["experience_years"]),
        "product_id": int(product["product_id"]),
        "product_group": product["product_group"],
        "quantity": quantity,
        "norm_minutes": norm_minutes,
        "actual_minutes": actual_minutes,
    })

df = pd.DataFrame(rows)

# немного реалистичного шума: пропуски в experience_years (как будто не всегда заполнено в проде)
missing_idx = RNG.choice(df.index, size=int(0.03 * len(df)), replace=False)
df.loc[missing_idx, "worker_experience_years"] = np.nan

df.to_csv("data/production_tasks.csv", index=False, encoding="utf-8-sig")
print(f"Сохранено {len(df)} строк в data/production_tasks.csv")
print(df.head())
