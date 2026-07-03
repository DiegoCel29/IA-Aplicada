"""
ag_core.py
==========
Modulo reutilizable con el Algoritmo Genetico (AG), Random Search y las
funciones de entrenamiento/evaluacion de la CNN.

Uso tipico dentro de cada notebook (uno por backbone: MobileNetV2, EfficientNet, ...):

    import ag_core as ag

    ag.configure(
        models={
            "MobileNetV2": {
                "fn": keras.applications.MobileNetV2,
                "preprocess": keras.applications.mobilenet_v2.preprocess_input,
            },
        },
        n_class=N_CLASS,
        seed=SEED,
        epochs_fitness=5,
        epochs_testeo=20,
    )

    population = ag.init_population(ag.POP_SIZE)
    ag_best, ag_curve, ag_stats = ag.genetic_algo(
        backbone="MobileNetV2",
        population=population,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        class_weight=CLASS_WEIGHT,
    )

Cada notebook solo se encarga de: cargar su dataset, definir el diccionario
`models` con su(s) backbone(s), y llamar a estas funciones. Los datos
(X_train, y_train, etc.) se pasan explicitamente a cada funcion en vez de
usar variables globales del notebook, para que el modulo sea reutilizable
sin importar que notebook lo importe.
"""

from __future__ import annotations

import json
import random
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

import keras
import torch
from keras import layers


# ==========================================================================
# 1. Configuracion global del AG (espacio de busqueda e hiperparametros)
# ==========================================================================
# Estos valores son los que definiste en AG-CNN_EX1. Puedes sobreescribirlos
# llamando a `configure(...)` antes de correr el AG, o reasignandolos
# directamente como ag_core.ALPHA = (...), etc.

ALPHA = (1e-4, 1e-2)
BATCH = (8, 16, 32, 64)
PHI = ("adam", "sgd", "rmsprop")
RHO = (0.0, 0.50)

UNIFORM_CROSS = ("batch", "phi")
BLEND_CROSSOVER = ("alpha", "rho")

POP_SIZE = 10
MUTATION_RATE = 0.15
GEN_MAX = 8
GEN_PAT = 3
K_SELECT = 3

EPOCHS_FITNESS = 5
EPOCHS_TESTEO = 20

N_CLASS = 3
SEED = 29

MODELS: dict = {}

OPTIMIZERS = {
    "adam": keras.optimizers.Adam,
    "sgd": keras.optimizers.SGD,
    "rmsprop": keras.optimizers.RMSprop,
}


def configure(
    models: dict,
    n_class: int = N_CLASS,
    seed: int = SEED,
    alpha=ALPHA,
    batch=BATCH,
    phi=PHI,
    rho=RHO,
    pop_size: int = POP_SIZE,
    mutation_rate: float = MUTATION_RATE,
    gen_max: int = GEN_MAX,
    gen_pat: int = GEN_PAT,
    k_select: int = K_SELECT,
    epochs_fitness: int = EPOCHS_FITNESS,
    epochs_testeo: int = EPOCHS_TESTEO,
):
    """Configura los parametros globales del AG para la sesion actual.

    Llamar una sola vez al inicio del notebook, despues de definir el/los
    backbone(s) en `models`.
    """
    global MODELS, N_CLASS, SEED, ALPHA, BATCH, PHI, RHO
    global POP_SIZE, MUTATION_RATE, GEN_MAX, GEN_PAT, K_SELECT
    global EPOCHS_FITNESS, EPOCHS_TESTEO

    MODELS = models
    N_CLASS = n_class
    SEED = seed
    ALPHA, BATCH, PHI, RHO = alpha, batch, phi, rho
    POP_SIZE = pop_size
    MUTATION_RATE = mutation_rate
    GEN_MAX = gen_max
    GEN_PAT = gen_pat
    K_SELECT = k_select
    EPOCHS_FITNESS = epochs_fitness
    EPOCHS_TESTEO = epochs_testeo

    random.seed(SEED)
    np.random.seed(SEED)


# ==========================================================================
# 2. Cromosoma e Individuo
# ==========================================================================

class HyperParams:
    def __init__(
        self,
        alpha: float | None = None,
        batch: int | None = None,
        phi: str | None = None,
        rho: float | None = None,
    ):
        self.alpha = alpha
        self.batch = batch
        self.phi = phi
        self.rho = rho

    def __getitem__(self, key: str):
        return getattr(self, key)

    def __setitem__(self, key: str, value):
        setattr(self, key, value)

    def keys(self):
        return ["alpha", "batch", "phi", "rho"]

    def copy(self):
        return deepcopy(self)

    def __repr__(self):
        return (f"HyperParams(alpha={self.alpha:.2e}, batch={self.batch}, "
                f"phi={self.phi}, rho={self.rho:.3f})")


class Individual:
    def __init__(self, chromosome: HyperParams | None = None):
        if chromosome is None:
            raise ValueError("Error: chromosome es nulo")
        self.chromosome = chromosome.copy()
        self.fitness: float = -1.0

    def set_fitness(self, value):
        if isinstance(value, np.ndarray):
            value = value.item()
        self.fitness = float(value)

    def crossover(self, other):
        chromosome1 = HyperParams()
        chromosome2 = HyperParams()

        for param in UNIFORM_CROSS:
            if random.uniform(0, 1) <= 0.5:
                chromosome1[param] = self.chromosome[param]
                chromosome2[param] = other.chromosome[param]
            else:
                chromosome1[param] = other.chromosome[param]
                chromosome2[param] = self.chromosome[param]

        for param in BLEND_CROSSOVER:
            a = random.uniform(0, 1)
            chromosome1[param] = a * self.chromosome[param] + (1 - a) * other.chromosome[param]
            chromosome2[param] = a * other.chromosome[param] + (1 - a) * self.chromosome[param]

        return Individual(chromosome1), Individual(chromosome2)

    def mutation(self):
        chromosome = self.chromosome.copy()
        for param in chromosome.keys():
            if random.uniform(0, 1) <= MUTATION_RATE:
                if param == "alpha":
                    chromosome[param] = random.uniform(*ALPHA)
                elif param == "batch":
                    chromosome[param] = random.choice(BATCH)
                elif param == "phi":
                    chromosome[param] = random.choice(PHI)
                elif param == "rho":
                    chromosome[param] = random.uniform(*RHO)
        return Individual(chromosome)


# ==========================================================================
# 3. Construccion y evaluacion del modelo CNN
# ==========================================================================

def build_model(backbone: str, hiperparams: HyperParams):
    base = MODELS[backbone]["fn"](
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    data_aug = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.05),
        ],
        name="augment",
    )

    inputs = keras.Input(shape=(224, 224, 3))
    x = data_aug(inputs)
    x = layers.Lambda(MODELS[backbone]["preprocess"])(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(hiperparams["rho"])(x)
    outputs = layers.Dense(N_CLASS, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=OPTIMIZERS[hiperparams["phi"]](learning_rate=hiperparams["alpha"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# --------------------------------------------------------------------
# Cache de fitness: evita reentrenar un cromosoma ya evaluado antes.
# La clave incluye el backbone, porque el mismo cromosoma puede dar
# fitness distinto segun la arquitectura.
# --------------------------------------------------------------------
FITNESS_CACHE: dict = {}


def chromosome_key(backbone: str, chromosome: HyperParams):
    # Se redondean los genes continuos para evitar claves distintas
    # por diferencias de decimales insignificantes.
    return (
        backbone,
        round(chromosome["alpha"], 6),
        chromosome["batch"],
        chromosome["phi"],
        round(chromosome["rho"], 3),
    )


def clear_fitness_cache():
    """Limpia el cache. Util si cambias el dataset o las epocas de fitness
    y quieres forzar que todo se reentrene."""
    FITNESS_CACHE.clear()


def get_fitness(
    backbone: str,
    chromosome: HyperParams,
    X_train, y_train,
    X_val, y_val,
    class_weight: dict,
    epochs: int | None = None,
    verbose_log: bool = True,
    use_cache: bool = True,
):
    key = chromosome_key(backbone, chromosome)

    if use_cache and key in FITNESS_CACHE:
        f1_macro = FITNESS_CACHE[key]
        if verbose_log:
            print(f"[cache] alpha={chromosome['alpha']:.2e} batch={chromosome['batch']:>2} "
                  f"phi={chromosome['phi']} rho={chromosome['rho']:.2f} | "
                  f"F1={f1_macro:.4f}  (sin reentrenar)")
        return f1_macro

    keras.backend.clear_session()

    model = build_model(backbone, chromosome)

    t0 = time.time()
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs or EPOCHS_FITNESS,
        batch_size=chromosome["batch"],
        class_weight=class_weight,
        verbose=0,
    )
    train_time = time.time() - t0

    y_prob = model.predict(X_val, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    f1_macro = f1_score(y_val, y_pred, average="macro")

    del model
    torch.cuda.empty_cache()

    if verbose_log:
        print(f"alpha={chromosome['alpha']:.2e} batch={chromosome['batch']:>2} "
              f"phi={chromosome['phi']} rho={chromosome['rho']:.2f} | "
              f"F1={f1_macro:.4f}  t={train_time:5.1f}s")

    if use_cache:
        FITNESS_CACHE[key] = f1_macro

    return f1_macro


def evaluate_test(
    backbone: str,
    chromosome: HyperParams,
    X_train, y_train,
    X_test, y_test,
    class_weight: dict,
    epochs: int | None = None,
):
    keras.backend.clear_session()

    model = build_model(backbone, chromosome)

    t0 = time.time()
    model.fit(
        X_train, y_train,
        epochs=epochs or EPOCHS_TESTEO,
        batch_size=chromosome["batch"],
        class_weight=class_weight,
        verbose=0,
    )
    train_time = time.time() - t0

    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    f1 = f1_score(y_test, y_pred, average="macro")
    acc = accuracy_score(y_test, y_pred)

    del model
    torch.cuda.empty_cache()

    return f1, acc, y_pred, train_time


# ==========================================================================
# 4. Poblacion, seleccion y AG / Random Search
# ==========================================================================

def init_population(size: int):
    population = []
    for _ in range(size):
        chromosome = (
            random.uniform(*ALPHA),
            random.choice(BATCH),
            random.choice(PHI),
            random.uniform(*RHO),
        )
        population.append(Individual(HyperParams(*chromosome)))
    return population


def eval_population(backbone, population, **fitness_kwargs):
    """fitness_kwargs: X_train, y_train, X_val, y_val, class_weight (y opcional epochs)."""
    for indiv in population:
        if indiv.fitness == -1:
            fitness = get_fitness(backbone, indiv.chromosome, **fitness_kwargs)
            indiv.set_fitness(fitness)


def select_parents(population, tournament_size):
    x1 = np.random.permutation(len(population))
    y1 = x1[:tournament_size]
    list_indiv = [population[i].fitness for i in y1]
    iParent1 = np.argmax(list_indiv)

    x2 = np.delete(x1, iParent1)
    x2 = np.random.permutation(x2)
    y2 = x2[:tournament_size]
    list_indiv = [population[i].fitness for i in y2]
    iParent2 = np.argmax(list_indiv)

    return population[y1[iParent1]], population[y2[iParent2]]


def select_survivors(population, offspring, num_survivors):
    combined = population + offspring
    combined.sort(key=lambda x: x.fitness, reverse=True)
    return combined[:num_survivors]


def genetic_algo(
    backbone,
    population,
    n_gen=None,
    p_gen=None,
    **fitness_kwargs,
):
    """fitness_kwargs: X_train, y_train, X_val, y_val, class_weight (y opcional epochs)."""
    n_gen = n_gen or GEN_MAX
    p_gen = p_gen or GEN_PAT
    pop_size = len(population)
    t_start = time.time()

    eval_population(backbone, population, **fitness_kwargs)

    best = sorted(population, key=lambda x: x.fitness, reverse=True)[0]
    best_fitness = [best.fitness]
    ag_stats = [{
        "gen": 0, "best": float(best.fitness),
        "mean": float(np.mean([i.fitness for i in population])),
        "std": float(np.std([i.fitness for i in population])),
    }]

    print(f"Poblacion inicial, mejor fitness = {best.fitness:.4f}")

    no_improve = 0
    TOL = 1e-4

    for gen in range(n_gen):
        mating_pool = [
            select_parents(population, K_SELECT)
            for _ in range(pop_size // 2)
        ]

        offspring = []
        for papa, mama in mating_pool:
            offspring.extend(papa.crossover(mama))
        offspring = [child.mutation() for child in offspring]

        eval_population(backbone, offspring, **fitness_kwargs)

        population = select_survivors(population, offspring, pop_size)
        best = sorted(population, key=lambda x: x.fitness, reverse=True)[0]
        best_fitness.append(best.fitness)

        f = [i.fitness for i in population]
        ag_stats.append({
            "gen": gen + 1, "best": float(best.fitness),
            "mean": float(np.mean(f)), "std": float(np.std(f)),
        })

        if best_fitness[-1] > best_fitness[-2] + TOL:
            no_improve = 0
            print(f"Generacion {gen + 1}, mejor fitness = {best_fitness[-1]:.4f}")
        else:
            no_improve += 1
            print(f"Generacion {gen + 1}, mejor fitness = {best_fitness[-1]:.4f} "
                  f"(sin mejora {no_improve}/{p_gen})")
            if no_improve >= p_gen:
                print(f"Early stopping: {p_gen} generaciones sin mejora.")
                break

    elapsed_time = time.time() - t_start
    print(f"Mejor individuo final, fitness = {best_fitness[-1]:.4f}  |  "
          f"Tiempo total AG = {elapsed_time:.1f}s")
    return best, best_fitness, ag_stats, elapsed_time


def random_search(backbone, n_evals, **fitness_kwargs):
    """fitness_kwargs: X_train, y_train, X_val, y_val, class_weight (y opcional epochs)."""
    t_start = time.time()
    population = init_population(n_evals)
    eval_population(backbone, population, **fitness_kwargs)

    best = None
    best_so_far = []
    for indiv in population:
        if best is None or indiv.fitness > best.fitness:
            best = indiv
        best_so_far.append(best.fitness)

    elapsed_time = time.time() - t_start
    print(f"Random Search | mejor fitness = {best.fitness:.4f}  |  "
          f"Tiempo total RS = {elapsed_time:.1f}s")
    return best, best_so_far, elapsed_time


# ==========================================================================
# 5. Guardado de resultados (para E3/E4: comparar entre arquitecturas)
# ==========================================================================

def _chrom_to_dict(c: HyperParams) -> dict:
    return {
        "alpha": float(c["alpha"]), "batch": int(c["batch"]),
        "phi": str(c["phi"]), "rho": float(c["rho"]),
    }


def save_experiment_results(
    backbone: str,
    out_dir: str | Path,
    default_indiv: Individual,
    ag_best: Individual, ag_curve: list, ag_stats: list,
    rs_best: Individual, rs_curve: list,
    test_results: dict,
    search_time: dict | None = None,
    extra_config: dict | None = None,
):
    """Guarda un JSON estandarizado por backbone en `out_dir/{backbone}.json`.

    `test_results` es un dict como:
        {"Algoritmo Genetico": {"f1": ..., "acc": ..., "train_time": ...}, "Random Search": {...}, "Default": {...}}

    `search_time` es un dict con el tiempo TOTAL de cada estrategia de busqueda
    (no el tiempo de entrenamiento final en test, sino el de encontrar el
    cromosoma ganador), por ejemplo:
        {"Algoritmo Genetico": ag_time, "Random Search": rs_time, "Default": default_time}

    Esto permite luego cargar todos los JSON de `out_dir` (uno por backbone)
    para hacer las comparativas E3 (mejor arquitectura) y E4 (vs literatura)
    sin tener que volver a correr nada.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "backbone": backbone,
        "seed": SEED,
        "config": {
            "POP_SIZE": POP_SIZE, "GEN_MAX": GEN_MAX, "GEN_PAT": GEN_PAT,
            "MUTATION_RATE": MUTATION_RATE, "K_SELECT": K_SELECT,
            "EPOCHS_FITNESS": EPOCHS_FITNESS, "EPOCHS_TESTEO": EPOCHS_TESTEO,
            **(extra_config or {}),
        },
        "default": {
            "chromosome": _chrom_to_dict(default_indiv.chromosome),
            "val_fitness": default_indiv.fitness,
        },
        "ag": {
            "chromosome": _chrom_to_dict(ag_best.chromosome),
            "val_fitness": ag_best.fitness,
            "curve": ag_curve,
            "stats": ag_stats,
        },
        "random_search": {
            "chromosome": _chrom_to_dict(rs_best.chromosome),
            "val_fitness": rs_best.fitness,
            "curve": rs_curve,
        },
        "test_results": test_results,
        "search_time_seconds": search_time or {},
    }

    out_path = out_dir / f"{backbone}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Resultados guardados en: {out_path}")
    return out_path


def load_all_results(out_dir: str | Path) -> dict:
    """Carga todos los JSON de resultados en out_dir, indexados por backbone.
    Util para E3 (comparar arquitecturas) y E4 (comparar con literatura)."""
    out_dir = Path(out_dir)
    results = {}
    for path in sorted(out_dir.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        results[data["backbone"]] = data
    return results
