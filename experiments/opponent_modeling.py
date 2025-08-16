import argparse
import os
import random
import csv
import subprocess
import time
from distutils.util import strtobool
from typing import List
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym.spaces import MultiDiscrete
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor, VecVideoRecorder
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="MicroRTSGridModeVecEnv",
                        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4,
                        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=50000000,
                        help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--prod-mode', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='run the script in produce mode and use wandb to log outputs')
    parser.add_argument('--capture-video', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='whether to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--wandb-project-name', type=str, default="gym-microrts",
                        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help="the entity (team) of wandb's project")

    # Algorithm specific arguments
    # ergänzt
    parser.add_argument('--epsilon-start', type=float, default=1.0,
                        help='Startwert für epsilon im epsilon-greedy Ansatz')
    parser.add_argument('--epsilon-final', type=float, default=0.02,
                        help='Minimaler epsilon-Wert')
    parser.add_argument('--epsilon-decay', type=int, default=100000,
                        help='Anzahl der Frames für linearen Epsilon-Zerfall')
    parser.add_argument('--sync-interval', type=int, default=1000,
                        help='Intervall in Frames zum Synchronisieren der Target-Netzwerke')
    # bis hier
    parser.add_argument('--partial-obs', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='if toggled, the game will have partial observability')
    parser.add_argument('--n-minibatch', type=int, default=4,
                        help='the number of mini batch')
    parser.add_argument('--num-bot-envs', type=int, default=1,
                        help='the number of bot game environment; 16 bot envs means 16 games')
    parser.add_argument('--num-selfplay-envs', type=int, default=0,
                        help='the number of self play envs; 16 self play envs means 8 games')
    parser.add_argument('--num-steps', type=int, default=256,
                        help='the number of steps per game environment')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='the discount factor gamma')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='the lambda for the general advantage estimation')
    parser.add_argument('--ent-coef', type=float, default=0.01,
                        help="coefficient of the entropy")
    parser.add_argument('--vf-coef', type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='the maximum norm for the gradient clipping')
    parser.add_argument('--clip-coef', type=float, default=0.1,
                        help="the surrogate clipping coefficient")
    parser.add_argument('--update-epochs', type=int, default=4,
                        help="the K epochs to update the policy")
    parser.add_argument('--kle-stop', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='If toggled, the policy updates will be early stopped w.r.t target-kl')
    parser.add_argument('--kle-rollback', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='If toggled, the policy updates will roll back to previous policy if KL exceeds target-kl')
    parser.add_argument('--target-kl', type=float, default=0.03,
                        help='the target-kl variable that is referred by --kl')
    parser.add_argument('--gae', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='Use GAE for advantage computation')
    parser.add_argument('--norm-adv', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help="Toggles advantages normalization")
    parser.add_argument('--anneal-lr', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument('--clip-vloss', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='Toggles whether or not to use a clipped loss for the value function, as per the paper.')
    parser.add_argument('--num-models', type=int, default=100,
                        help='the number of models saved')
    parser.add_argument('--max-eval-workers', type=int, default=4,
                        help='the maximum number of eval workers (skips evaluation when set to 0)')
    parser.add_argument('--train-maps', nargs='+', default=["maps/8x8/basesWorkers8x8.xml"],
                        help='the list of maps used during training')
    parser.add_argument('--eval-maps', nargs='+', default=["maps/8x8/basesWorkers8x8.xml"],
                        help='the list of maps used during evaluation')
    parser.add_argument('--max-steps', type=int, default=2000,
                        help='maximale Anzahl Schritte pro Spiel')
    parser.add_argument('--warmup-frames', type=int, default=100000,
                        help="Anzahl der Schritte mit ausschließlich Exploration")
    parser.add_argument("--buffer-memory", type=int, default=1000000,
                        help="Größe des speichers für den Replay Buffer")
    parser.add_argument("--save-network", type=int, default=300_000,
                        help="Wie häufig das Netz gespeichert werden soll")
    parser.add_argument('--sequence_length', type=int, default=4,
                        help='Sequenzlänge Rekkurenten Netz, bei 1 kein RNN')
    parser.add_argument('--eval_interval', type=int, default=50_000,
                        help='Wie häufig evaluiert wird')
    parser.add_argument('--num_iterations', type=int, default=10_000,
                        help='Number of iterations before Treaining ends')


    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())
    args.num_envs = args.num_selfplay_envs + args.num_bot_envs
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.n_minibatch)
    args.num_updates = args.total_timesteps // args.batch_size
    args.save_frequency = max(1, int(args.num_updates // args.num_models))
    # fmt: on
    return args

class MicroRTSStatsRecorder(VecEnvWrapper):
    """Nimmt eine Vektorisierte Umgebung und fügt Auswertungstools ein"""

    def __init__(self, env, gamma=0.99) -> None:
        super().__init__(env)
        self.gamma = gamma  # gamma ist unser discount faktor

    def reset(self):
        obs = self.venv.reset()
        self.raw_rewards = [[] for _ in range(self.num_envs)]
        self.ts = np.zeros(self.num_envs, dtype=np.float32)
        self.raw_discount_rewards = [[] for _ in range(self.num_envs)]
        return obs

    def step_wait(self):
        """obs reward, dones werden unverändert zurückgegeben und nur infos in newinfos ungewandeld
        """
        obs, rews, dones, infos = self.venv.step_wait()  # observation
        newinfos = list(infos[:])
        for i in range(len(dones)):
            self.raw_rewards[i] += [infos[i]["raw_rewards"]]
            self.raw_discount_rewards[i] += [
                (self.gamma ** self.ts[i])
                * np.concatenate((infos[i]["raw_rewards"], infos[i]["raw_rewards"].sum()), axis=None)
            ]
            self.ts[i] += 1
            if dones[i]:  # wenn Episode zu Ende
                info = infos[i].copy()
                raw_returns = np.array(self.raw_rewards[i]).sum(0)
                raw_names = [str(rf) for rf in self.rfs]
                raw_discount_returns = np.array(self.raw_discount_rewards[i]).sum(0)
                raw_discount_names = ["discounted_" + str(rf) for rf in self.rfs] + ["discounted"]
                info["microrts_stats"] = dict(zip(raw_names, raw_returns))
                info["microrts_stats"].update(dict(zip(raw_discount_names, raw_discount_returns)))
                self.raw_rewards[i] = []
                self.raw_discount_rewards[i] = []
                self.ts[i] = 0
                newinfos[i] = info
        return obs, rews, dones, newinfos


class OpponentActionNet(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        c, h, w = input_shape

        self.input_height = h
        self.input_width = w

        self.encoder = nn.Sequential(
            nn.Conv2d(c, c * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(c * 2, c * 2, kernel_size=3, padding=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out((c, h, w))

        #  +2 für (x, y) Position
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size + 2, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        """
        4 move
        4 harvest
        4 return
        4*7 produce direction
        49 attack dir"""
        self.out = nn.Linear(512,89)

    def forward(self, x, unit_pos=None):
        # x: [B, C, H, W]
        batch_size = x.size(0)
        x = self.encoder(x).reshape(batch_size, -1)

        if unit_pos is None:
            # Standard: keine Positionsinfo
            pos = torch.zeros((batch_size, 2), device=x.device)
        else:
            # Normalisiere Position auf [0, 1]
            norm_x = unit_pos[:, 0].float() / (self.input_width - 1)
            norm_y = unit_pos[:, 1].float() / (self.input_height - 1)
            pos = torch.stack([norm_x, norm_y], dim=1)  # [B, 2]

        x = torch.cat([x, pos], dim=1)
        fc_out = self.fc(x)
        out = self.out(fc_out)
        return out

    def _get_conv_out(self, shape):
        o = self.encoder(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

class OpponentModeling:
    def __init__(self,env, state ,net, device):
        self.env=env
        self.net=net
        self.device=device
        #self.state=self.env.state
        self.predictions = []
        self.unit_idx = 0 # zählt Anzahl der Units für die ein Label vorliegt

    def qval_to_action(self, q_val: int) -> np.ndarray:
        """
        Wandelt einen diskreten Q-Wert (0–88) in ein Aktionsarray mit 7 Feldern um.

        Rückgabe: Array mit:
        [action_type, move_dir, harvest_dir, return_dir, produce_dir, produce_type, attack_idx]
        """
        if not (0 <= q_val <= 88):
            raise ValueError("q_val muss im Bereich 0–88 liegen.")

        action = np.zeros(7, dtype=np.int32)

        if 0 <= q_val < 4:
            action[0] = 1  # action_type: move
            action[1] = q_val

        elif 4 <= q_val < 8:
            action[0] = 2  # harvest
            action[2] = q_val - 4

        elif 8 <= q_val < 12:
            action[0] = 3  # return
            action[3] = q_val - 8

        elif 12 <= q_val < 40:
            action[0] = 4  # produce
            produce_idx = q_val - 12
            action[5] = produce_idx // 4  # produce_type
            action[4] = produce_idx % 4  # produce_dir

        elif 40 <= q_val <= 88:
            action[0] = 5  # attack
            action[6] = q_val - 40

        return action

    def action_to_qval(self, single_action):
        """
        single_action: list or tensor with 7 integers
        [action_type, move_dir, harvest_dir, return_dir, produce_dir, produce_type, attack_idx]
        """
        a_type = single_action[0]
        if a_type == 0:
            return 0
        elif a_type == 1:
            return 0 + single_action[1]
        elif a_type == 2:
            return 4 + single_action[2]
        elif a_type == 3:
            return 8 + single_action[3]
        elif a_type == 4:
            return 12 + single_action[5] * 4 + single_action[4]
        elif a_type == 5:
            return 40 + single_action[6]
        else:
            raise ValueError(f"Ungültiger action_type: {a_type}")

    def convert_78_to_89_mask(self, mask_78):
        """Konvertiert 78-dim Aktionmaske → 89-diskrete Aktionsmaske"""
        assert mask_78.shape[0] == 78
        if isinstance(mask_78, np.ndarray):
            mask_78 = torch.from_numpy(mask_78).bool()
        m89 = torch.zeros(89, dtype=torch.bool, device=mask_78.device)

        # Action types
        is_move = mask_78[1]
        is_harvest = mask_78[2]
        is_return = mask_78[3]
        is_produce = mask_78[4]
        is_attack = mask_78[5]

        # MOVE direction (mask_78[6–9]) → m89[0–3]
        if is_move:
            for i in range(4):
                if mask_78[6 + i]:
                    m89[i] = True

        # HARVEST direction (mask_78[10–13]) → m89[4–7]
        if is_harvest:
            for i in range(4):
                if mask_78[10 + i]:
                    m89[4 + i] = True

        # RETURN direction (mask_78[14–17]) → m89[8–11]
        if is_return:
            for i in range(4):
                if mask_78[14 + i]:
                    m89[8 + i] = True

        # PRODUCE (type [22–28] × direction [18–21]) → m89[12–39]
        if is_produce:
            for type_idx in range(7):  # 7 Typen: ressource ... ranged
                if mask_78[22 + type_idx]:
                    for dir_idx in range(4):  # 4 Richtungen: N, E, S, W
                        if mask_78[18 + dir_idx]:
                            m89[12 + 4 * type_idx + dir_idx] = True

        if is_attack:
        # Attack (mask_78[29–77]) → m89[40–88]
            for i in range(49):
                if mask_78[29 + i]:
                    m89[40 + i] = True

        return m89

    def predict(self):
        net = self.net
        device = self.device
        raw_masks_np = self.env.venv.venv.get_action_mask()
        raw_masks = torch.from_numpy(raw_masks_np).to(device=device).bool()  # [num_envs, H*W, 78]
        _, h, w, _ = self.state.shape
        num_envs = self.env.num_envs

        state_v = torch.tensor(self.state.transpose(0, 3, 1, 2), dtype=torch.float32, device=device)
        self.num_units = 0
        for env_i in range(num_envs):
            for i in range(h):
                for j in range(w):
                    if self.state[env_i, i, j, 12] == 1 and self.state[env_i, i, j, 21] == 1:
                        self.unit_idx += 1
                        self.num_units += 1

                        #State, Unit Pos und Action Mask holen
                        state_v_single = state_v[env_i:env_i + 1]
                        unit_pos = torch.tensor([[j, i]], dtype=torch.float32, device=device)
                        flat_idx = i * h + j
                        cell_mask = self.convert_78_to_89_mask(raw_masks[env_i, flat_idx])

                        # Q vals berechnen
                        q_vals_v = net(state_v_single, unit_pos=unit_pos)[0]  # [89]
                        # Q vals maskieren
                        masked_q_vals = q_vals_v.masked_fill(~cell_mask, -1e9)
                        # Maskierte Q-Vals
                        self.predictions.append({
                            "unit_idx": self.unit_idx,
                            "env_idx": env_i,
                            "unit_pos": (i, j),
                            #"pred_action_type": action_type,
                            "q_vals": masked_q_vals
                        })


    def observe(self):
        for i in range(self.num_units):
            pred = self.predictions[-self.num_units + i]   #holt sich die zuletzt gespeicherte Unit
            env_idx = pred["env_idx"]
            x, y = pred["unit_pos"]
            observed_action_type = None

            for j in range(6):                              #ließt die current Action Plane 21-26
                if self.state[env_idx, x, y, 21 + j] == 1:
                    observed_action_type = j
                    break

            pred["observed_action_type"] = observed_action_type     #speichert diese in pred

    def calc_action_type_loss(self):
        logits_list = []
        labels_list = []
        for pred in self.predictions:
            obs_type = pred.get("observed_action_type")
            logits = pred.get("q_vals")  # Shape: [89]

            if obs_type is None or logits is None:
                continue

            device = logits.device
            action_type_logits = torch.zeros(6, device=device)
            action_type_logits[1] = logits[0:4].max()  # Move
            action_type_logits[2] = logits[4:8].max()  # Harvest
            action_type_logits[3] = logits[8:12].max()  # Return
            action_type_logits[4] = logits[12:40].max()  # Produce
            action_type_logits[5] = logits[40:].max()  # Attack
            action_type_logits[0] = logits.min() - 1 # nicht verwendet
            logits_list.append(action_type_logits.unsqueeze(0))  # [1, 6]
            labels_list.append(torch.tensor([obs_type], device=device))

        if len(logits_list) == 0:
            return None

        x = torch.cat(logits_list, dim=0)  # [Batch, 6]
        y = torch.cat(labels_list, dim=0)  # [Batch]
        loss = nn.CrossEntropyLoss()(x, y)

        return loss


"""Initialisierung"""

"""
1)  Initialisieren
2)  Agent laden
3)  OpponentModeling Predict
4)  Agent Step
5)  OpponentModeling Observe
6)  Wiederhole 3-5 bis Batchgröße erreicht ist
7)  Aktualisiere Opponent Modeling Netz
8)  Evaluiere
8)  Wiederhole 3-8 bis 
"""

def Training(agent, batch_size: int, optimizer=None, learning_rate: float = 1e-4, eval_ratio: float = 0.2):
    """
    Führt genau EINE Trainings-Aktualisierung für das Opponent-Modeling-Netz durch.
    Ablauf:
        1) OpponentModeling.predict() auf aktuellem Zustand
        2) Agenten-Schritt (env.step)
        3) OpponentModeling.observe() auf neuem Zustand
        4) Wiederholen bis mindestens 'batch_size' gelabelte Units gesammelt sind
        5) Backprop auf Basis von calc_action_type_loss()
        6) Optional: kurze Evaluation (ohne Gradienten) über eval_ratio*batch_size Units

    Parameter
    ---------
    agent : SB3-ähnliches Modell mit .predict(...) und zugehöriger VecEnv (model.get_env() oder model.env)
    batch_size : int
        Anzahl gelabelter Einheiten (Units), die für EINEN Optimizer-Step gesammelt werden.
    optimizer : torch.optim.Optimizer | None
        Falls None, wird (einmalig) ein Adam-Optimizer am agent hinterlegt.
    learning_rate : float
        Lernrate für den (neu) angelegten oder übergebenen Optimizer.
    eval_ratio : float
        Anteil der Batchgröße, der für eine kurze Evaluation nach dem Update verwendet wird (0.0 → keine Eval).

    Returns
    -------
    dict mit Schlüsseln:
        "loss": float | None
        "train_acc": float | None
        "eval_acc": float | None
    """
    # 1) Env & Obs holen
    env = agent.get_env() if hasattr(agent, "get_env") and agent.get_env() is not None else getattr(agent, "env", None)
    if env is None:
        raise RuntimeError("Konnte keine Env am Agent finden (weder .get_env() noch .env).")

    obs = env.reset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) OpponentModel / Netz initialisieren (einmalig am Agent cachen)
    if not hasattr(agent, "_opponent_modeling"):
        # Input-Shape (C, H, W) für das Netz aus der Beobachtung ableiten
        if len(obs.shape) != 4:
            raise RuntimeError(f"Erwarte VecEnv-Obs mit Shape [num_envs, H, W, C], bekam: {obs.shape}")
        _, H, W, C = obs.shape
        net = OpponentActionNet((C, H, W)).to(device)
        opponent_model = OpponentModeling(env=env, state=obs, net=net, device=device)
        agent._opponent_modeling = opponent_model

        if optimizer is None:
            optimizer = optim.Adam(opponent_model.net.parameters(), lr=learning_rate)
        agent._opponent_optimizer = optimizer
    else:
        opponent_model = agent._opponent_modeling
        if optimizer is None:
            optimizer = getattr(agent, "_opponent_optimizer", None)
            if optimizer is None:
                optimizer = optim.Adam(opponent_model.net.parameters(), lr=learning_rate)
                agent._opponent_optimizer = optimizer
        else:
            # übergebene LR ggf. setzen
            for g in optimizer.param_groups:
                g["lr"] = learning_rate

    # Kleine Hilfsfunktion: logits für Aktionstyp aus 89er-Q-Werten bauen
    #atl->action_type_logits
    def _action_type_logits_from_qvals(q_vals: torch.Tensor) -> torch.Tensor:
        atl = torch.zeros(6, device=q_vals.device)
        atl[1] = q_vals[0:4].max()      # Move
        atl[2] = q_vals[4:8].max()      # Harvest
        atl[3] = q_vals[8:12].max()     # Return
        atl[4] = q_vals[12:40].max()    # Produce
        atl[5] = q_vals[40:].max()      # Attack
        atl[0] = -1e9                   # "no-op"/nicht verwendet
        return atl

    # 3) Daten für EIN Update sammeln
    opponent_model.predictions.clear()
    units_collected = 0

    while units_collected < batch_size:
        # Vor dem Step: Vorhersagen für aktuelle Gegner-Units erzeugen
        opponent_model.state = obs
        opponent_model.predict()

        # Agenten-Aktion ausführen (Fallback: random action, falls predict fehlschlägt)
        try:
            actions, _ = agent.play_step(epsilon=0.5)
        except Exception:
            actions = env.action_space.sample()
        print(f"obs.shape: {obs.shape}")
        print(f"actions.shape: {actions.shape}")
        obs, rewards, dones, infos = env.step(actions)

        # Nach dem Step: tatsächlich beobachtete Aktionstypen einsammeln
        opponent_model.state = obs
        opponent_model.observe()

        units_collected += opponent_model.num_units

        # VecEnv setzt automatisch zurück nur weiterlaufen

    # 4) Loss berechnen und Optimizer-Step
    loss = opponent_model.calc_action_type_loss()
    train_acc = None
    if loss is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(opponent_model.net.parameters(), 1.0)
        optimizer.step()

        # Training-Accuracy über die gesammelte Batch
        correct, total = 0, 0
        with torch.no_grad():
            for pred in opponent_model.predictions:
                obs_type = pred.get("observed_action_type")
                q_vals = pred.get("q_vals")
                if obs_type is None or q_vals is None:
                    continue
                pred_type = int(torch.argmax(_action_type_logits_from_qvals(q_vals)).item())
                correct += int(pred_type == obs_type)
                total += 1
        train_acc = (correct / total) if total > 0 else None

    # 5) Kurze Evaluation (ohne Gradienten), falls gewünscht
    eval_acc = None
    eval_units_target = int(max(0, eval_ratio) * batch_size)
    if eval_units_target > 0:
        opponent_model.predictions.clear()
        collected, correct, total = 0, 0, 0
        with torch.no_grad():
            while collected < eval_units_target:
                opponent_model.state = obs
                opponent_model.predict()
                try:
                    actions, _ = agent.predict(obs, deterministic=True)
                except Exception:
                    actions = env.action_space.sample()

                obs, rewards, dones, infos = env.step(actions)
                opponent_model.state = obs
                opponent_model.observe()

                # nur die zuletzt hinzugefügten Vorhersagen der aktuellen Schrittfolge auswerten
                for pred in opponent_model.predictions[-opponent_model.num_units:]:
                    ot = pred.get("observed_action_type")
                    qv = pred.get("q_vals")
                    if ot is None or qv is None:
                        continue
                    pred_type = int(torch.argmax(_action_type_logits_from_qvals(qv)).item())
                    correct += int(pred_type == ot)
                    total += 1

                collected += opponent_model.num_units

        eval_acc = (correct / total) if total > 0 else None

    return {
        "loss": (float(loss.item()) if loss is not None else None),
        "train_acc": (float(train_acc) if train_acc is not None else None),
        "eval_acc": (float(eval_acc) if eval_acc is not None else None),
    }

class RandomAgent:
    def __init__(self, env):
        self.env = env
        self.num_envs = env.num_envs
        self.last_obs = self.env.reset()

    def predict(self, obs=None, deterministic=False):
        # Gibt zufällige Aktionen für alle Envs zurück
        if obs is None:
            obs = self.last_obs
        actions = self.env.action_space.sample()
        return actions, None

    def play_step(self, epsilon=0.0):
        # Nimmt zufällige Aktionen, führt einen Schritt aus
        actions = self.env.action_space.sample()
        next_obs, rewards, dones, infos = self.env.step(actions)
        envs.venv.venv.render(mode="human")
        self.last_obs = next_obs
        return actions, rewards

    def get_env(self):
        return self.env

if __name__ == "__main__":
    print(">>> Argumentparser wird initialisiert")
    args = parse_args()

    print(f"Save frequency: {args.save_frequency}")

    # TRY NOT TO MODIFY: setup the environment
    experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.prod_mode:
        import wandb

        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            # sync_tensorboard=True,
            config=vars(args),
            name=experiment_name,
            monitor_gym=True,
            save_code=True,
        )
        wandb.tensorboard.patch(save=False)
    writer = SummaryWriter(f"runs/{experiment_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]))
    )

    # TRY NOT TO MODIFY: seeding
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    print(f"Device: {device}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    reward_weights = np.array([50.0, 3.0, 3.0, 0.0, 5.0, 1.0])
    num_envs = args.num_bot_envs
    num_each = num_envs // 4
    envs = MicroRTSGridModeVecEnv(
        num_selfplay_envs=args.num_selfplay_envs,
        num_bot_envs=args.num_bot_envs,
        partial_obs=args.partial_obs,
        max_steps=2000,
        render_theme=2,
        ai2s=(
            [microrts_ai.passiveAI for _ in range(num_each)] +
            [microrts_ai.workerRushAI for _ in range(num_each)] +
            [microrts_ai.lightRushAI for _ in range(num_each)] +
            [microrts_ai.coacAI for _ in range(num_envs - 3 * num_each)]
        ),
        # microrts_ai.lightRushAI coacAI passiveAi
        map_paths=[args.train_maps[0]],
        reward_weight=reward_weights, #Win, Ressource, ProduceWorker, Produce Building, Attack, ProduceCombat Unit, (auskommentiert closer to enemy base)
        cycle_maps=args.train_maps,
    )
    envs = MicroRTSStatsRecorder(envs, args.gamma)
    envs = VecMonitor(envs)
    if args.capture_video:
        envs = VecVideoRecorder(
            envs, f"videos/{experiment_name}", record_video_trigger=lambda x: x % 100000 == 0, video_length=2000
        )
    assert isinstance(envs.action_space, MultiDiscrete), "only MultiDiscrete action space is supported"

    eval_executor = None
    if args.max_eval_workers > 0:
        from concurrent.futures import ThreadPoolExecutor

        eval_executor = ThreadPoolExecutor(max_workers=args.max_eval_workers, thread_name_prefix="league-eval-")

    agent = RandomAgent(envs)


    # CSV-Datei initialisieren
    model_path = f'./{args.exp_name}/'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    csv_path = f'./{args.exp_name}/{args.exp_name}_csv.csv'
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["iteration", "loss", "train_acc", "eval_acc"])
    # evalierung initialisieren
    best_eval_mean = 0
    eval_window = []
    save_every = args.save_network  # Intervall für regelmäßiges Speichern
    # Training durchführen
    print(args.batch_size)
    for iteration in range(args.num_iterations):
        result = Training(agent, batch_size=args.batch_size)
        # Ergebnisse drucken
        print(f"[{iteration}] Loss: {result['loss']:.4f}, "
              f"Train-Acc: {result['train_acc']:.2%}, "
              f"Eval-Acc: {result['eval_acc']:.2%}")

        # Ergebnisse in CSV speichern
        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                iteration,
                result['loss'],
                result['train_acc'],
                result['eval_acc']
            ])
        # Gleitender Durchschnitt der letzten 10 Evaluationen
        eval_window.append(result['eval_acc'])
        if len(eval_window) > 10:
            eval_window.pop(0)
        eval_mean = sum(eval_window) / len(eval_window)
        # Modell bei verbessertem Eval-Durchschnitt speichern
        if eval_mean > best_eval_mean:
            best_eval_mean = eval_mean
            torch.save(agent._opponent_modeling.net.state_dict(), f"./{args.exp_name}/best_model.pt")
            print(f"Modell mit verbessertem Eval-Durchschnitt ({eval_mean:.4f}) gespeichert.")

        # Regelmäßiges Speichern
        if iteration % save_every == 0 and iteration > 0:
            torch.save(agent._opponent_modeling.net.state_dict(), f"{args.exp_name}/model_iter_{iteration}.pt")
            print(f"Modell bei Iteration {iteration} gespeichert.")

