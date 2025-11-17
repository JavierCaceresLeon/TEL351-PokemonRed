# Advanced Gym Agents

This package introduces three purpose-built agents to benchmark against the baseline PPO model. Each agent ships with a tailored observation pipeline, transition model, reward function, and inference stack so they can be compared scientifically.

| Agent | Focus | Core Ideas |
|-------|-------|------------|
| `CombatApexAgent` | Never lose a battle and end fights with the highest margin possible | Damage-difference rewards, GRU dynamics head that predicts HP deltas, Transformer feature extractor for turn memory, CVaR-style risk penalty for potential deaths |
| `PuzzleSpeedAgent` | Reach each gym leader with minimal puzzles/steps | Graph-based latent state summarizing switch/panel states, occupancy-penalizing reward, differentiable shortest-path heuristic via differentiable value iteration network |
| `HybridSageAgent` | Balance both while planning consumable usage for long campaigns | Pareto front reward blending, adaptive temperature on action distribution, shared world model with dual critics that reason about puzzle and combat returns |

The sections below document how the new agents modify the environment contract.

## Common Notation

Let `s_t` be the raw memory snapshot available through `RedGymEnv`, `o_t` the constructed observation given to the policy, `a_t` an action, and `r_t` the shaped reward. `hp_t` and `hp^opp_t` are normalized party and opponent hit-points, `d_t` is cumulative damage dealt since the last reset, `m_t` is the current map index, and `c_t` counts explored coordinates.

## Combat Apex Agent

### Reward

```
r_t = 4 * (d_t - d_{t-1}) + 6 * (hp_t - hp_{t-1}) - 3 * max(0, hp_{t-1} - hp_t)
       + 25 * 1[battle_won] - 50 * 1[pokemon_fainted] - 100 * 1[loss]
```

The wrapper also adds a CVaR penalty by tracking the left tail of the damage distribution and subtracting `lambda * VaR_0.1` when the agent plays recklessly.

### Observation

Adds a `battle_features` vector consisting of:

1. Party HP ratios and status flags for each slot (6 * 3 scalars).
2. Opponent HP, level, and type encoding (one-hot over 15 types).
3. Turn tempo indicators (last four action deltas, timeout counters).
4. Damage velocity (EMA over the previous 8 steps).

The features are passed through a small Transformer encoder before being concatenated with the base screen stack.

### Transition / Inference

A GRU-based dynamics head `f_theta(h_{t-1}, a_{t-1}, battle_features_t)` predicts the next HP deltas and opponent faint probability. The auxiliary loss is
```
L_dyn = ||hp_{t} - \hat{hp_t}||_2^2 + CE(1[battle_won_t], \hat{p}_t)
```
and is optimized jointly with PPO via a custom callback, enabling the policy to reason about impending knockouts.

## Puzzle Speed Agent

### Reward

```
r_t = -0.05 * steps + 15 * 1[new_panel_state]
       + 30 * 1[leader_room_entered] - 25 * 1[non_leader_battle]
       + 40 * 1[badge_increment]
```

A differentiable shortest-path heuristic (`V_t`) from a learned value-iteration module further shapes the reward by adding `+ alpha * (V_{t-1} - V_t)` to encourage progress along optimal puzzle routes.

### Observation

Augments the base dict with `puzzle_features`:

- Global coordinate patch from the exploration map projected onto a 16x16 occupancy grid.
- Binary switches representing puzzle elements (statues, floor panels, trash cans) read directly from RAM.
- Estimated geodesic distance to the leader, derived from the learned graph.

### Transition / Inference

Uses a value-iteration inspired `PuzzleGraphTransitionModel` that embeds the walkable tiles as nodes and applies a differentiable Bellman backup for five iterations. The produced potential field feeds both the reward bonus and the policy features, allowing model-based lookahead without leaving Gym API compatibility.

## Hybrid Sage Agent

### Reward

The hybrid agent keeps two running advantages `A^combat_t` and `A^puzzle_t` and computes a Pareto-optimal blend using a dynamic weight `w_t` based on consumable usage and current badge progress:

```
w_t = sigmoid( beta_0 + beta_1 * badge_count - beta_2 * item_usage_rate )
r_t = w_t * A^combat_t + (1 - w_t) * A^puzzle_t - 0.1 * item_overuse_t
```

A KL-targeted entropy term keeps exploration calibrated for long campaigns.

### Observation

Adds both `battle_features` and `puzzle_features`, plus a `resource_vector` summarizing items, badges, and fainted pokemon. A lightweight cross-attention block lets the policy fuse the modalities.

### Transition / Inference

Builds a shared latent dynamics model with two heads (combat/puzzle) and minimizes a multi-task loss
```
L = L_dyn^combat + L_dyn^puzzle + \lambda * D_{KL}(q(z_t) || p(z_t | s_t))
```

which mirrors recent latent-imagination methods. During inference, the policy queries the latent planner for five imagined steps to bias action logits toward long-term survivability.

## How to Use

The `advanced_agents.train_agents` module exposes helper functions:

```python
from advanced_agents.train_agents import train_combat_apex, train_puzzle_speed, train_hybrid_sage

train_combat_apex(total_timesteps=5_000_000)
```

Each helper builds the necessary wrappers, policies, and auxiliary models before delegating to Stable Baselines3. See the docstrings inside each module for concrete implementation details.

## Recommended Evaluation Workflow

1. **Entrena cada agente** con `train_*` y conserva el checkpoint en `advanced_agents/runs/<agent>/model.zip`.
1. **Convierte el checkpoint** a un path accesible por `gym_scenarios/evaluate_agents.py`.
1. **Ejecuta el comparador** especificando el modelo base (PPO original) y el agente avanzado:

```bash
python gym_scenarios/evaluate_agents.py \
       --baseline runs/ppo_original/model.zip \
       --improved advanced_agents/runs/combat_apex/model.zip \
       --scenario-set gym_scenarios/scenarios.json
```

1. **Analiza los reportes JSON** generados por el evaluador para comparar recompensa total, pasos, y métricas específicas de cada gimnasio.

Como los nuevos agentes registran métricas auxiliares (daño, potencial de rompecabezas, uso de ítems), los resultados exportan campos adicionales con prefijo `combat_aux_` y `puzzle_aux_` que puedes graficar desde `test_metrics_system.py` o tus propios notebooks.
