import time

import numpy as np


# Improved Orca Predation Algorithm (IOPA) Starting Line No. 72
def Proposed(pop, obj_fn, VRmin, VRmax, max_iters):
    pop_size, dim = pop.shape[0], pop.shape[1]
    # algorithm hyperparameters
    a = 1.0
    d = 1.0
    F = 2.0,
    b = 0.5
    c = 0.5
    w = 0.4,
    m = 0.1
    s = 0.1
    e = 0.2,
    p1 = 0.1
    rng_seed = 0,
    rng = np.random.default_rng(rng_seed)

    vel1 = np.zeros_like(pop)  # V^t_chase,1
    vel2 = np.zeros_like(pop)  # V^t_chase,2

    def clip(x):
        return np.minimum(np.maximum(x, VRmin), VRmax)

    def S_t(t):
        # Adaptive attack speed S(t) (Eq. 21–23)
        return 1.0 - (t / max(1, max_iters))

    # initial fitness
    fitness = obj_fn(pop)
    best_idx = np.argmin(fitness)
    best_pos = pop[best_idx].copy()
    best_val = fitness[best_idx]
    Convergence_curve = np.zeros((max_iters, 1))

    ct = time.time()
    for t in range(max_iters):
        Xbest = best_pos.copy()
        Xsocial = pop.mean(axis=0)  # Eq. (16)
        Mt = pop.mean(axis=0)  # Eq. (13)

        # -----------------------------
        # Driving phase
        # -----------------------------
        for i in range(pop_size):
            xti = pop[i]

            # Eq. (13): V^t_chase,1,i
            v1 = a * (d * Xbest - F * (b * Mt + c * xti)) + w * vel1[i]
            # Eq. (15): + m*(Xbest - xti)
            v1 += m * (Xbest - xti)
            # Eq. (16): + s*(Xsocial - xti)
            v1 += s * (Xsocial - xti)

            # Eq. (14): V^t_chase,2,i
            v2 = e * (Xbest - xti) + w * vel2[i]

            vel1[i], vel2[i] = v1, v2

        # -----------------------------
        # Encircling phase (bubble-net)
        # -----------------------------
        # Eq. (18): B^t
        Bt = np.sum([Xbest - pop[j] for j in range(pop_size)], axis=0) / max(1, pop_size)

        new_positions = pop.copy()
        for i in range(pop_size):
            CurrentFit = fitness[i]
            BestFit = np.min(fitness)
            WorstFit = np.max(fitness)
            r = (CurrentFit + BestFit) / (CurrentFit + BestFit + WorstFit)

            # Technique 1: chase using Eq. (13)+(15)+(16)
            xtchase_1 = pop[i] + vel1[i]

            # Technique 2: chase using Eq. (14)
            xtchase_2 = pop[i] + vel2[i]

            # Technique 3: Eq. (17)
            u = rng.uniform(-1, 1)
            idxs = rng.choice(pop_size, size=min(3, pop_size), replace=False)
            xtd1, xtd2, xtd3 = pop[idxs[0]], pop[idxs[1]], pop[idxs[2]]
            xtchase_3 = xtd1 + u * (xtd2 - xtd3) + b * Bt

            if r < p1:
                candidate = xtchase_1
            elif r < 2 * p1:
                candidate = xtchase_2
            else:
                candidate = xtchase_3

            candidate = clip(candidate)
            cand_f = obj_fn(candidate)

            # Eq. (19)–(20): greedy replacement
            if cand_f[i] < fitness[i]:
                new_positions[i] = candidate[i]
                fitness[i] = cand_f[i]

        pop = new_positions

        # -----------------------------
        # Attacking phase
        # -----------------------------
        S_val = S_t(t)
        for i in range(pop_size):
            # Eq. (21): V^t_attack,1,i
            idxs = rng.choice(pop_size, size=min(4, pop_size), replace=False)
            xts = np.mean(pop[idxs], axis=0)
            xtchase_i = pop[i]
            Vtattack1 = xts - xtchase_i * S_val

            # Eq. (22): V^t_attack,2,i
            idxs2 = rng.choice(pop_size, size=min(3, pop_size), replace=False)
            xts2 = np.mean(pop[idxs2], axis=0)
            Vtattack2 = xts2 - pop[i] * S_val

            # Eq. (23): x^t_attack,i
            g1, g2 = rng.uniform(0, 1), rng.uniform(0, 1)
            new_pos = xtchase_i + g1 * Vtattack1 + g2 * Vtattack2
            new_pos = clip(new_pos)

            val_new = obj_fn(new_pos)
            if val_new[i] < fitness[i]:
                pop[i] = new_pos[i]
                fitness[i] = val_new[i]

        # update global best
        cur_best_idx = np.argmin(fitness)
        if fitness[cur_best_idx] < best_val:
            best_val = fitness[cur_best_idx]
            best_pos = pop[cur_best_idx].copy()
            Convergence_curve[t] = best_val
        best_val = Convergence_curve[max_iters - 1][0]
        ct = time.time() - ct

    return best_val, Convergence_curve, best_pos, ct
