import random

def generate_instance(n, K, d=4, conflict_density=0.3, seed=42):
    random.seed(seed)
    tasks = [f'T{i}' for i in range(n)]
    conflicts = [(i,j) for i in range(n) for j in range(i+1,n) if random.random() < conflict_density]
    cap = [32, 128, 8, 6.0]
    resources = [[random.uniform(1, cap[d]//(n//K+1)) for d in range(4)] for _ in range(n)]
    capacities = [cap[:] for _ in range(K)]
    windows = [(lo := random.randint(0, K-2), random.randint(lo+1, K-1)) for _ in range(n)]
    weights = [random.uniform(1, 10) for _ in range(n)]
    return dict(tasks=tasks, conflicts=conflicts, resources=resources, capacities=capacities, windows=windows, weights=weights, K=K)