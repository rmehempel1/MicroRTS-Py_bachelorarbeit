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
from wandb.cli.cli import agent


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
    parser.add_argument('--num-bot-envs', type=int, default=4,
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


def to_scalar(x):
    """Konvertiert NumPy-Array, Tensor oder float/int zu float."""
    try:
        if hasattr(x, 'mean'):
            return float(x.mean())
        return float(x)
    except Exception as e:
        print(f"[WARN] to_scalar failed for {x}: {e}")
        return 0.0

import csv

def log_episode_to_csv(
    csv_path: str,
    episode_idx: int,
    frame_idx: int,
    reward: float,
    mean_reward: float,
    eval_reward: float,
    loss: float,
    epsilon: float,
    dauer: float,
    reward_counts: dict,
    reward_names: list
):
    """
    Schreibt eine abgeschlossene Episode in eine CSV-Datei.
    """
    row = [
        episode_idx,
        frame_idx,
        reward,
        mean_reward,
        eval_reward,
        loss if loss is not None else "",
        epsilon,
        dauer
    ] + [reward_counts.get(name, 0) for name in reward_names]

    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists or os.stat(csv_path).st_size == 0:
            writer.writerow(["episode", "frame", "reward","mean_reward","eval_reward", "loss", "epsilon", "dauer"] + reward_names)
        writer.writerow(row)

def evaluate(agent, envs, device, num_episodes=1):
    total_reward = 0.0

    for _ in range(num_episodes):
        agent._reset()  # Setzt internen Zustand zurück
        done = False
        while not done:
            result = agent.play_eval(device=device, epsilon=0.0)
            done = result["done"]
            if done:
                total_reward += result["reward"]

    return total_reward / num_episodes


def get_headwise_action_mask(env, actions_shape, head_config):
    """
    Erzeugt eine Aktionsmaske pro Head (move, attack, produce, etc.).

    Args:
        env: MicroRTS-Environment mit `get_action_mask()`-Methode
        actions_shape: Form des Action-Tensors, z. B. [B, H, W, 7]
        head_config: Dict wie in DQN verwendet, mit "type_id" und "indices"

    Returns:
        mask_dict: Dict mit bool-Masken für jeden Headname (B, H, W, A)
    """
    action_mask = env.get_action_mask()  # [B, H, W, 6, max_param]
    B, H, W, num_components, max_param = action_mask.shape
    mask_dict = {}

    for name, config in head_config.items():
        type_id = config["type_id"]
        indices = config["indices"]

        # move, attack, etc. haben typischerweise eine Parametermaske an indices[1]
        # produce hat zwei: indices[1] = dir, indices[2] = unit
        if name == "produce":
            dir_mask = action_mask[:, :, :, type_id, indices[1]]  # z. B. Richtung (4er)
            unit_mask = action_mask[:, :, :, type_id, indices[2]]  # z. B. Einheitentyp (2er)
            mask_dict[name] = (dir_mask, unit_mask)
        else:
            param_mask = action_mask[:, :, :, type_id, indices[1]]
            mask_dict[name] = param_mask  # [B, H, W, num_options]

    return mask_dict


class ExperienceBuffer:
    def __init__(self, capacity):
        """
        Initialisiert den Replay Buffer mit einer festen Kapazität.
        Neuere Einträge überschreiben automatisch die ältesten (FIFO).
        """
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        """
        Gibt die aktuelle Anzahl gespeicherter Transitionen zurück.
        """
        return len(self.buffer)

    def append(self, experience):
        """
        Fügt eine neue Erfahrung zum Buffer hinzu.
        Erwartet ein Tuple: (state, action, reward, done, next_state)
        """
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        states, actions, action_taken_grid, rewards, dones, next_states = zip(*batch)

        states = np.array(states)
        next_states = np.array(next_states)
        actions = np.stack(actions)  # → shape: (B, 448)

        # Rekonstruiere optional [B, H, W, 7]
        B = actions.shape[0]
        # print(actions.shape)
        grid_size = int(np.sqrt(actions.shape[1] // 7))
        # print("Grid_size", grid_size)
        actions = actions.reshape(B, grid_size, grid_size, 7)

        return (
            states,
            actions,
            action_taken_grid,
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.uint8),
            next_states
        )

def add_positional_encoding(state):
    """
    Fügt jedem Zustand einen normierten x/y-Positionskanal hinzu.
    Erwartet: state.shape = (B, H, W, C)
    Gibt zurück: (B, H, W, C+2)
    """
    B, H, W, _ = state.shape

    # Normierte Koordinaten
    x_coords = np.linspace(0, 1, W)
    y_coords = np.linspace(0, 1, H)

    # Raster erzeugen
    x_grid = np.tile(x_coords, (H, 1))        # (H, W)
    y_grid = np.tile(y_coords[:, None], (1, W))  # (H, W)

    # Zu Shape (B, H, W, 1) erweitern
    x_grid = np.broadcast_to(x_grid, (B, H, W))
    y_grid = np.broadcast_to(y_grid, (B, H, W))

    x_grid = x_grid[..., np.newaxis]  # (B, H, W, 1)
    y_grid = y_grid[..., np.newaxis]  # (B, H, W, 1)

    # Anhängen
    return np.concatenate([state, x_grid, y_grid], axis=-1)  # → (B, H, W, C+2)


class MovementHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.encoder_decision = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.decision_head = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1),
            nn.Softmax(dim=1)  # Softmax über [no-op, move]
        )

        self.encoder_dir = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.dir_head = nn.Sequential(
            nn.Conv2d(64, 4, kernel_size=1),
            nn.Softmax(dim=1)  # Softmax über Richtungen 0–3
        )

    def forward(self, x):
        x_decision = self.encoder_decision(x)
        decision_probs = self.decision_head(x_decision)  # [B, 2, H, W]

        x_dir = self.encoder_dir(x)
        dir_probs = self.dir_head(x_dir)  # [B, 4, H, W]

        return decision_probs, dir_probs

class AttackHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.encoder_decision = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Entscheidung: angreifen oder nicht → 2 Klassen
        self.decision_head = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1),
            nn.Softmax(dim=1)  # Softmax über [no-op, attack]
        )

        self.encoder_target = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Zielauswahl (Index aus 49 möglichen Zellen = 7x7 Grid)
        self.target_head = nn.Sequential(
            nn.Conv2d(64, 49, kernel_size=1),
            nn.Softmax(dim=1)  # Softmax über mögliche Zielzellen
        )

    def forward(self, x):
        x_decision = self.encoder_decision(x)
        decision_probs = self.decision_head(x_decision)  # [B, 2, H, W]

        x_target = self.encoder_target(x)
        target_probs = self.target_head(x_target)        # [B, 49, H, W]

        return decision_probs, target_probs



class ProduceHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # Entscheidung: produzieren oder nicht (2 Klassen)
        self.encoder_decision = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decision_head = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1),
            nn.Softmax(dim=1)  # [no-op, produce]
        )

        # Richtung (4 Richtungen)
        self.encoder_dir = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.dir_head = nn.Sequential(
            nn.Conv2d(64, 4, kernel_size=1),
            nn.Softmax(dim=1)
        )

        # Typauswahl (UnitTypeTable size, z. B. 7)
        self.encoder_type = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.type_head = nn.Sequential(
            nn.Conv2d(64, 7, kernel_size=1),  # ggf. anpassen, falls andere Unit-Anzahl
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        decision_probs = self.decision_head(self.encoder_decision(x))  # [B, 2, H, W]
        dir_probs = self.dir_head(self.encoder_dir(x))                 # [B, 4, H, W]
        type_probs = self.type_head(self.encoder_type(x))             # [B, 7, H, W]

        return decision_probs, dir_probs, type_probs



def sync_target_heads(policy_heads, target_heads):
    for name in policy_heads:
        # 1. Parameter auslesen
        params = policy_heads[name].state_dict()
        # 2. In Zielnetz laden
        target_heads[name].load_state_dict(params)




class Agent:
    def __init__(self, env, exp_buffer, device="cpu"):
        """
        Initialisiert den Agenten mit Zugriff auf die Umgebung und den Replay Buffer.
        """

        self.device = device
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()
        self.in_channels=29+2

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU()
        ).to(self.device)

        self.movement_head = MovementHead(in_channels=self.in_channels).to(self.device)
        self.harvest_head = MovementHead(in_channels=self.in_channels).to(self.device)
        self.return_head = MovementHead(in_channels=self.in_channels).to(self.device)
        self.produce_head = ProduceHead(in_channels=self.in_channels).to(self.device)
        self.attack_head = AttackHead(in_channels=self.in_channels).to(self.device)

        self.heads = {
            "attack": self.attack_head,
            "harvest": self.harvest_head,
            "return": self.return_head,
            "move": self.movement_head,
            "produce": self.produce_head,
        }

        self.head_config = {
            "attack": {
                "type_id": 5,
                "indices": (0, 6), #indices: relevanten action Components
                "classes": (2, 4),  #2-> decision 4->richtungen
                "param_indices": {
                    "decision": 0,
                    "direction": 6
                }
            },
            "harvest": {
                "type_id": 2,
                "indices": (0, 2),
                "classes": (2, 4),
                "param_indices": {
                    "decision": 0,
                    "direction": 2
                }
            },
            "return": {
                "type_id": 3,
                "indices": (0, 3),
                "classes": (2, 4),
                "param_indices": {
                    "decision": 0,
                    "direction": 3
                }
            },
            "produce": {
                "type_id": 4,
                "indices": (0, 4, 5),
                "classes": (2, 4, 7),
                "param_indices": {
                    "decision": 0,
                    "unit_type": 5
                }
            },
            "move": {
                "type_id": 1,
                "indices": (0, 1),
                "classes": (2, 4),
                "param_indices": {
                    "decision": 0,
                    "direction": 1
                }
            }
        }

    def _get_structured_action_masks(self, device):
        """
        Erstellt eine strukturierte Aktionsmaske, welche angibt welche Aktionen auf welchem Grid gültig sind
        Nach Aktionstypen aufgeschlüsselt und in PyTorch-Tensoren überführt
        """
        raw_masks = self.env.venv.venv.get_action_mask()  # [num_envs, b*h, 78]
        grid_size = int(np.sqrt(raw_masks.shape[1]))

        def reshape_and_convert(mask, channels):
            mask = mask.reshape(self.env.num_envs, grid_size, grid_size, channels)
            return torch.tensor(mask, dtype=torch.float32, device=device).permute(0, 3, 1, 2)

        return {
            "action_type": reshape_and_convert(raw_masks[:, :, 0:6], 6),
            "move_dir": reshape_and_convert(raw_masks[:, :, 6:10], 4),
            "harvest_dir": reshape_and_convert(raw_masks[:, :, 10:14], 4),
            "return_dir": reshape_and_convert(raw_masks[:, :, 14:18], 4),
            "produce_dir": reshape_and_convert(raw_masks[:, :, 18:22], 4),
            "produce_type": reshape_and_convert(raw_masks[:, :, 22:29], 7),
            "attack_dir": reshape_and_convert(raw_masks[:, :, 29:78], 49),
        }

    def get_action_type_grid(self, structured_masks,
                             attack_decision,
                             harvest_decision,
                             return_decision,
                             produce_decision,
                             move_decision):
        """
        Priorisierte Auswahl der Action Types basierend auf gültigen Masken.
        1: move, 2: harvest, 3: return, 4: produce, 5: attack
        Default ist 0 (no-op)
        """
        action_type_mask = structured_masks["action_type"]


        



        E, H, W = attack_decision.shape
        action_type_grid = np.zeros((E, H, W), dtype=np.int32)

        for i in range(E):
            for j in range(H):
                for k in range(W):
                    valid_types = action_type_mask[i, :, j, k].cpu().numpy()
                    """
                    produce_decision[i,j,k]=1
                    attack_decision[i, j, k]=1
                    harvest_decision[i, j, k]=1
                    return_decision[i, j, k]=1
                    move_decision[i, j, k]=1
                    """
                    #print(i,j,k)
                    #print("valid types,", valid_types)
                    action_type_grid[i, j, k] = 1

                    if len(valid_types) != 6:
                        raise ValueError(f"Ungültige Länge der Masken: {len(valid_types)} an Pos. [{i},{j},{k}]")
                    """
                    print("valid", valid_types)
                    print("Move", move_decision[i, j, k])
                    """
                    if valid_types[5] and attack_decision[i, j, k] == 1:
                        action_type_grid[i, j, k] = 5
                    elif valid_types[2] and harvest_decision[i, j, k] == 1:
                        action_type_grid[i, j, k] = 2
                    elif valid_types[3] and return_decision[i, j, k] == 1:
                        action_type_grid[i, j, k] = 3
                    elif valid_types[4] and produce_decision[i, j, k] == 1:
                        action_type_grid[i, j, k] = 4
                    elif valid_types[1] and move_decision[i, j, k] == 1:

                        action_type_grid[i, j, k] = 1
                    # else action-> entfernt
                    #print("action_type_grid", action_type_grid[i,j,k])
        return action_type_grid

    def merge_actions(self,
            action_type_grid,  # (E, H, W) – Priorisierte Aktionsauswahl
            attack_params=None,  # (H, W, 2)
            harvest_mask=None,  # (H, W) – bool
            return_mask=None,  # (H, W) – bool
            produce_params=None,  # (H, W, 1)
            produce_type=None,
            move_params=None  # (H, W, 1)
    ):
        """
        Erstellt einen flachen Aktionsvektor aus den Einzel-Head-Ausgaben.
        action_type_ bestimmt, welche Aktion pro Grid aktiv ist.
        Die restlichen Parameter werden gesetzt, wenn die Aktion aktiv ist.
        Format: 7 Einträge pro Zelle, wie in MicroRTS erwartet.
        """
        #print("action_type_grid", action_type_grid.shape)
        E, H, W = action_type_grid.shape
        #print("E:", E)
        full_action = np.zeros((E, H, W, 7), dtype=np.int32)
        # print("produce_params.shape:", produce_params.shape)
        for i in range(E):
            for j in range(H):
                for k in range(W):
                    a_type = action_type_grid[i, j, k]  # holt sich den action type
                    #print(f"Pos [{i},{j},{k}] a_type={a_type}")
                    full_action[i, j, k, 0] = a_type  # action type eintragen

                    # Aktion ausführen, andere Parameterfelder auf 0
                    if a_type == 5:  # action_type=5 -> Attack
                        # print("attack_params.shape:", attack_params.shape)
                        # print("Beispielwert:", attack_params[i, j, k])
                        full_action[i, j, k, 6] = attack_params[i, j, k]

                    elif a_type == 2:
                        full_action[i, j, k, 2] = harvest_mask[i, j, k]

                    elif a_type == 3:
                        full_action[i, j, k, 3] = return_mask[i, j, k]

                    elif a_type == 4:
                        #print(produce_params.dtype, produce_params.shape)
                        #print("produce_params, produce_type, shape", produce_params.shape, produce_type.shape)
                        #print("produce_params, produce_type", produce_params[i, j, k], produce_type[i, j, k])
                        full_action[i, j, k, 4] = produce_params[i, j, k]
                        full_action[i, j, k, 5] = produce_type[i, j, k]
                        # print("Produce")

                    elif a_type == 1 :
                        full_action[i, j, k, 1] = move_params[i, j, k]


        return full_action.reshape(E, H * W * 7)

    def _reset(self):
        """
        Startet eine neue Episode und setzt interne Zustände zurück.
        """
        self.state = self.env.reset()
        self.total_reward = 0.0

    @torch.no_grad()
    def play_eval(self,device="cpu",epsilon=0.5):


        full_mask = self.env.venv.venv.get_action_mask()
        #print("Full Mask:   ", full_mask.shape)
        enhanced_state = add_positional_encoding(self.state)
        state_v = torch.tensor(enhanced_state, dtype=torch.float32, device=self.device).permute(0, 3, 1, 2)
        #print("state v:     ", state_v.shape)

        """Jeder Kopf muss alle seine maximal Möglichen Aktionen machen, diese einzeln. Die beste Aktion an Merge 
        schicken, welcher die Gesamtaktion ausführt
        """
        # Berechne strukturierte Aktionsmasken
        masks = self._get_structured_action_masks( device=self.device)
        #print("masks:       ", masks["action_type"].shape)


        # Attack
        attack_dec = self.attack_head.encoder_decision(state_v)
        #print("attack_dec:      ", attack_dec.shape)
        attack_decision_logits = self.attack_head.decision_head[0](attack_dec)
        #print("attack_decision_logits", attack_decision_logits.shape)
        attack_allowed_mask = masks["action_type"][:, 5]  # [B, H, W]
        #print("attack_allowed mask:     ", attack_allowed_mask)
        attack_decision_logits[:, 1] = attack_decision_logits[:, 1].masked_fill(attack_allowed_mask == 0, float("-inf"))
        #print("attack_decision_logits:      ", attack_decision_logits.shape)
        attack_mask = attack_decision_logits.argmax(dim=1).cpu().numpy()
        #print("attack mask:     ", attack_mask.shape)

        att_dir = self.attack_head.encoder_target(state_v)
        #print("att_dir:     ", att_dir)
        attack_dir_logits = self.attack_head.target_head[0](att_dir)
        #print("attack_dir_logits:       ", attack_dir_logits)
        attack_dir_masked = attack_dir_logits.masked_fill(masks["attack_dir"] == 0, float("-inf"))
        #print("attack_dir_masked:       ", attack_dir_masked)
        attack_param = attack_dir_masked.argmax(dim=1).cpu().numpy()
        #print("attack_param:    ", attack_param)

        # Move
        # Entscheidung
        move_dec = self.movement_head.encoder_decision(state_v)
        move_decision_logits = self.movement_head.decision_head[0](move_dec)
        move_allowed_mask = masks["action_type"][:, 1]  # move erlaubt?
        move_decision_logits[:, 1] = move_decision_logits[:, 1].masked_fill(move_allowed_mask == 0, float("-inf"))
        move_mask = move_decision_logits.argmax(dim=1).cpu().numpy()

        # Richtung
        move_dir = self.movement_head.encoder_dir(state_v)
        move_dir_logits = self.movement_head.dir_head[0](move_dir)
        move_dir_masked = move_dir_logits.masked_fill(masks["move_dir"] == 0, float("-inf"))
        move_param = move_dir_masked.argmax(dim=1).cpu().numpy()

        # Harvest
        # Entscheidung
        harv_dec = self.harvest_head.encoder_decision(state_v)
        harvest_decision_logits = self.harvest_head.decision_head[0](harv_dec)
        harvest_allowed_mask = masks["action_type"][:, 2]  # Index 2 = harvest
        harvest_decision_logits[:, 1] = harvest_decision_logits[:, 1].masked_fill(harvest_allowed_mask == 0,
                                                                                  float("-inf"))
        harvest_mask = harvest_decision_logits.argmax(dim=1).cpu().numpy()

        # Richtung
        harv_dir = self.harvest_head.encoder_dir(state_v)
        harvest_dir_logits = self.harvest_head.dir_head[0](harv_dir)
        harvest_dir_masked = harvest_dir_logits.masked_fill(masks["harvest_dir"] == 0, float("-inf"))
        harvest_param = harvest_dir_masked.argmax(dim=1).cpu().numpy()

        # Return
        # Entscheidung
        ret_dec = self.return_head.encoder_decision(state_v)
        return_decision_logits = self.return_head.decision_head[0](ret_dec)
        return_allowed_mask = masks["action_type"][:, 3]
        return_decision_logits[:, 1] = return_decision_logits[:, 1].masked_fill(return_allowed_mask == 0, float("-inf"))
        return_mask = return_decision_logits.argmax(dim=1).cpu().numpy()

        # Richtung
        ret_dir = self.return_head.encoder_dir(state_v)
        return_dir_logits = self.return_head.dir_head[0](ret_dir)
        return_dir_masked = return_dir_logits.masked_fill(masks["return_dir"] == 0, float("-inf"))
        return_param = return_dir_masked.argmax(dim=1).cpu().numpy()

        # Produce
        # Entscheidung
        prod_dec = self.produce_head.encoder_decision(state_v)
        produce_decision_logits = self.produce_head.decision_head[0](prod_dec)
        produce_allowed_mask = masks["action_type"][:, 4]
        produce_decision_logits[:, 1] = produce_decision_logits[:, 1].masked_fill(produce_allowed_mask == 0,
                                                                                  float("-inf"))
        produce_mask = produce_decision_logits.argmax(dim=1).cpu().numpy()

        # Richtung
        prod_dir = self.produce_head.encoder_dir(state_v)
        produce_dir_logits = self.produce_head.dir_head[0](prod_dir)
        produce_dir_masked = produce_dir_logits.masked_fill(masks["produce_dir"] == 0, float("-inf"))
        produce_param = produce_dir_masked.argmax(dim=1).cpu().numpy()

        # Typ
        prod_type = self.produce_head.encoder_type(state_v)
        produce_type_logits = self.produce_head.type_head[0](prod_type)
        produce_type_masked = produce_type_logits.masked_fill(masks["produce_type"] == 0, float("-inf"))
        produce_type = produce_type_masked.argmax(dim=1).cpu().numpy()

        # Führe Teilaktion zur Gesamtaktion zusammen
        action_type_grid = self.get_action_type_grid(masks,
                                                     attack_mask,
                                                     harvest_mask,
                                                     return_mask,
                                                     produce_mask,
                                                     move_mask)
        action_taken_grid = action_type_grid
        #print("action taken", action_taken_grid)

        """

        print("attack_mask.shape:", attack_mask.shape)
        print("move_mask.shape:", move_mask.shape)
        print("state_v.shape:", state_v.shape)
        """
        #print("vor merge",action_type_grid.shape, attack_dec.shape)
        action = self.merge_actions(
            action_type_grid,
            attack_param,
            harvest_param,
            return_param,
            produce_param,  # zuerst!
            produce_type,  # danach!
            move_param
        )
        #print("play eval aktion:", action)
        # print("doppelcheck", action.shape)

        # Führe Aktion aus

        torch.tensor(self.env.venv.venv.get_action_mask(), dtype=torch.float32)

        new_state, reward, is_done, infos = self.env.step(action)
        #print("Infos (raw):", infos)
        #print("Keys:", list(infos[0].keys()))

        self.total_reward += reward
        return {"done": np.any(is_done), "reward": self.total_reward[0], "infos": infos[0]}

        # print("action.shape before storing:", action.shape)





    @torch.no_grad()
    def play_step(self, epsilon=0.0, device="cpu"):
        """
                for idx in range(64):
            valid = np.sum(raw_mask[0, idx])
            if valid > 0:
                #print(f"Index {idx} gültig mit {valid} Aktionen")

        for i in range(8):
            for j in range(8):
                base = (i * 8 + j) * 6
                print(f"Zelle ({i},{j}) → Index {base}")
        """

        """
        Führt einen Schritt im Environment aus:
        - Wählt eine Aktion mittels ε-greedy Strategie
        - Führt Aktion im Environment aus
        - Speichert Transition im Replay Buffer
        - Rückgabe: Gesamt-Reward bei Episodenende, sonst None
        """
        done_reward = None
        full_mask = self.env.venv.venv.get_action_mask()
        # ε-greedy Aktionsauswahl
        if np.random.random() < epsilon:
            raw_masks = self.env.venv.venv.get_action_mask()  # [num_envs, H*W, 78]
            grid_size = int(np.sqrt(raw_masks.shape[1]))

            def sample_valid(mask_2d):
                valid_indices = np.where(mask_2d)[0]
                if len(valid_indices) == 0:
                    return 0  # Fallback auf 0
                return np.random.choice(valid_indices)

            action = np.zeros((self.env.num_envs, grid_size, grid_size, 7), dtype=np.int32)
            action_taken_grid = np.zeros((self.env.num_envs, grid_size, grid_size), dtype=np.int32)
            for env_i in range(self.env.num_envs):
                for idx in range(grid_size * grid_size):
                    cell_mask = raw_masks[env_i, idx]
                    i, j = divmod(idx, grid_size)

                    a_type = sample_valid(cell_mask[0:6])
                    action[env_i, i, j, 0] = a_type
                    action_taken_grid[env_i, i, j] = a_type
                    if a_type == 1:  # Move
                        action[env_i, i, j, 1] = sample_valid(cell_mask[6:10])
                    elif a_type == 2:  # Harvest
                        action[env_i, i, j, 2] = sample_valid(cell_mask[10:14])
                    elif a_type == 3:  # Return
                        action[env_i, i, j, 3] = sample_valid(cell_mask[14:18])
                    elif a_type == 4:  # Produce
                        action[env_i, i, j, 4] = sample_valid(cell_mask[18:22])
                        action[env_i, i, j, 5] = sample_valid(cell_mask[22:29])
                    elif a_type == 5:  # Attack
                        action[env_i, i, j, 6] = sample_valid(cell_mask[29:78])

            action = action.reshape(self.env.num_envs, -1)
            # Führe Aktion aus
            torch.tensor(self.env.venv.venv.get_action_mask(), dtype=torch.float32)
            new_state, reward, is_done, infos = self.env.step(action)
            self.total_reward += reward
        else:
            # Zustand vorbereiten für Netzwerkeingabe
            full_mask = self.env.venv.venv.get_action_mask()
            enhanced_state = add_positional_encoding(self.state)
            state_v = torch.tensor(enhanced_state, dtype=torch.float32, device=self.device).permute(0, 3, 1, 2)

            """Jeder Kopf muss alle seine maximal Möglichen Aktionen machen, diese einzeln. Die beste Aktion an Merge 
            schicken, welcher die Gesamtaktion ausführt
            """
            # Berechne strukturierte Aktionsmasken
            masks = self._get_structured_action_masks(device=self.device)
            # print("state_v shaoe", state_v.shape)

            # Attack
            attack_dec = self.attack_head.encoder_decision(state_v)
            attack_decision_logits = self.attack_head.decision_head[0](attack_dec)
            attack_allowed_mask = masks["action_type"][:, 5]  # [B, H, W]
            attack_decision_logits[:, 1] = attack_decision_logits[:, 1].masked_fill(attack_allowed_mask == 0,
                                                                                    float("-inf"))
            attack_mask = attack_decision_logits.argmax(dim=1).cpu().numpy()

            att_dir = self.attack_head.encoder_target(state_v)
            attack_dir_logits = self.attack_head.target_head[0](att_dir)
            attack_dir_masked = attack_dir_logits.masked_fill(masks["attack_dir"] == 0, float("-inf"))
            attack_param = attack_dir_masked.argmax(dim=1).cpu().numpy()

            # Move
            # Entscheidung
            move_dec = self.movement_head.encoder_decision(state_v)
            move_decision_logits = self.movement_head.decision_head[0](move_dec)
            move_allowed_mask = masks["action_type"][:, 1]  # move erlaubt?
            move_decision_logits[:, 1] = move_decision_logits[:, 1].masked_fill(move_allowed_mask == 0, float("-inf"))
            move_mask = move_decision_logits.argmax(dim=1).cpu().numpy()

            # Richtung
            move_dir = self.movement_head.encoder_dir(state_v)
            move_dir_logits = self.movement_head.dir_head[0](move_dir)
            move_dir_masked = move_dir_logits.masked_fill(masks["move_dir"] == 0, float("-inf"))
            move_param = move_dir_masked.argmax(dim=1).cpu().numpy()

            # Harvest
            # Entscheidung
            harv_dec = self.harvest_head.encoder_decision(state_v)
            harvest_decision_logits = self.harvest_head.decision_head[0](harv_dec)
            harvest_allowed_mask = masks["action_type"][:, 2]  # Index 2 = harvest
            harvest_decision_logits[:, 1] = harvest_decision_logits[:, 1].masked_fill(harvest_allowed_mask == 0,
                                                                                      float("-inf"))
            harvest_mask = harvest_decision_logits.argmax(dim=1).cpu().numpy()

            # Richtung
            harv_dir = self.harvest_head.encoder_dir(state_v)
            harvest_dir_logits = self.harvest_head.dir_head[0](harv_dir)
            harvest_dir_masked = harvest_dir_logits.masked_fill(masks["harvest_dir"] == 0, float("-inf"))
            harvest_param = harvest_dir_masked.argmax(dim=1).cpu().numpy()

            # Return
            # Entscheidung
            ret_dec = self.return_head.encoder_decision(state_v)
            return_decision_logits = self.return_head.decision_head[0](ret_dec)
            return_allowed_mask = masks["action_type"][:, 3]
            return_decision_logits[:, 1] = return_decision_logits[:, 1].masked_fill(return_allowed_mask == 0,
                                                                                    float("-inf"))
            return_mask = return_decision_logits.argmax(dim=1).cpu().numpy()

            # Richtung
            ret_dir = self.return_head.encoder_dir(state_v)
            return_dir_logits = self.return_head.dir_head[0](ret_dir)
            return_dir_masked = return_dir_logits.masked_fill(masks["return_dir"] == 0, float("-inf"))
            return_param = return_dir_masked.argmax(dim=1).cpu().numpy()

            # Produce
            # Entscheidung
            prod_dec = self.produce_head.encoder_decision(state_v)
            produce_decision_logits = self.produce_head.decision_head[0](prod_dec)
            produce_allowed_mask = masks["action_type"][:, 4]
            produce_decision_logits[:, 1] = produce_decision_logits[:, 1].masked_fill(produce_allowed_mask == 0,
                                                                                      float("-inf"))
            produce_mask = produce_decision_logits.argmax(dim=1).cpu().numpy()

            # Richtung
            prod_dir = self.produce_head.encoder_dir(state_v)
            produce_dir_logits = self.produce_head.dir_head[0](prod_dir)
            produce_dir_masked = produce_dir_logits.masked_fill(masks["produce_dir"] == 0, float("-inf"))
            produce_param = produce_dir_masked.argmax(dim=1).cpu().numpy()

            # Typ
            prod_type = self.produce_head.encoder_type(state_v)
            produce_type_logits = self.produce_head.type_head[0](prod_type)
            produce_type_masked = produce_type_logits.masked_fill(masks["produce_type"] == 0, float("-inf"))
            produce_type = produce_type_masked.argmax(dim=1).cpu().numpy()

            # Führe Teilaktion zur Gesamtaktion zusammen
            action_type_grid = self.get_action_type_grid(masks,
                                                         attack_mask,
                                                         harvest_mask,
                                                         return_mask,
                                                         produce_mask,
                                                         move_mask)
            action_taken_grid = action_type_grid


            """

            print("attack_mask.shape:", attack_mask.shape)
            print("move_mask.shape:", move_mask.shape)
            print("state_v.shape:", state_v.shape)
            """
            # print("vor merge",action_type_grid.shape, attack_dec.shape)
            action = self.merge_actions(
                action_type_grid,
                attack_param,
                harvest_param,
                return_param,
                produce_param,  # zuerst!
                produce_type,  # danach!
                move_param
            )
            # print("play eval aktion:", action)
            # print("doppelcheck", action.shape)

            # Führe Aktion aus
            torch.tensor(self.env.venv.venv.get_action_mask(), dtype=torch.float32)
            new_state, reward, is_done, infos = self.env.step(action)


            self.total_reward += reward

        # print("action.shape before storing:", action.shape)

        for env_i in range(self.env.num_envs):
            self.exp_buffer.append((
                self.state[env_i],
                action[env_i],  # → shape: (448,)
                action_taken_grid[env_i],
                reward[env_i],
                is_done[env_i],
                new_state[env_i]
            ))

        self.state = new_state
        if np.any(is_done):
            done_reward = self.total_reward
            self._reset()
            return {"done": True, "reward": done_reward[0], "infos": infos[0]}
        return {"done": False, "reward": reward[0], "infos": infos[0]}

    def calc_loss(self, batch, target_heads, gamma=0.99):
        device = self.device
        states, actions, action_taken_grid, rewards, dones, next_states = batch
        states = add_positional_encoding(states)
        next_states = add_positional_encoding(next_states)


        # Tensor-Konvertierung und Formatierung
        states_t = torch.tensor(states, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device).view(-1, 1, 1)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=device).view(-1, 1, 1)
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        action_taken_grid = torch.tensor(np.array(action_taken_grid), dtype=torch.long, device=device)

        q_preds, q_tgts = [], []

        for name, head in self.heads.items():
            cfg = self.head_config[name]
            type_id = cfg["type_id"]
            mask = (action_taken_grid == type_id)

            if not mask.any():
                continue

            out = head(states_t)
            tgt = target_heads[name](next_states_t)

            if name == "produce":
                dir_idx, type_idx = cfg["indices"][1], cfg["indices"][2]

                act_dirs = actions[:, :, :, dir_idx]
                act_types = actions[:, :, :, type_idx]
                act_dec = (action_taken_grid == 4).long()

                q_dir = out[1].gather(1, act_dirs.unsqueeze(1)).squeeze(1)
                q_dec = out[0].gather(1, act_dec.unsqueeze(1)).squeeze(1)
                q_type = out[2].gather(1, act_types.unsqueeze(1)).squeeze(1)

                tgt_dec = tgt[0].max(1).values
                tgt_dir = tgt[1].max(1).values
                tgt_type = tgt[2].max(1).values

                q_preds += [q_dec[mask], q_dir[mask], q_type[mask]]
                q_tgts += [
                    (rewards_t.expand_as(tgt_dec) + gamma * tgt_dec * (1.0 - dones_t.expand_as(tgt_dec)))[mask],
                    (rewards_t.expand_as(tgt_dir) + gamma * tgt_dir * (1.0 - dones_t.expand_as(tgt_dir)))[mask],
                    (rewards_t.expand_as(tgt_type) + gamma * tgt_type * (1.0 - dones_t.expand_as(tgt_type)))[mask],
                ]

            else:
                param_idx = cfg["indices"][1]
                act_param = actions[:, :, :, param_idx]
                act_dec = (action_taken_grid == cfg["indices"][0]).long()

                q_dec = out[0].gather(1, act_dec.unsqueeze(1)).squeeze(1)
                q = out[1].gather(1, act_param.unsqueeze(1)).squeeze(1)

                tgt_q_dec = tgt[0].max(1).values
                tgt_q = tgt[1].max(1).values

                q_preds += [q_dec[mask], q[mask]]
                q_tgts += [
                    (rewards_t.expand_as(tgt_q_dec) + gamma * tgt_q_dec * (1.0 - dones_t.expand_as(tgt_q_dec)))[mask],
                    (rewards_t.expand_as(tgt_q) + gamma * tgt_q * (1.0 - dones_t.expand_as(tgt_q)))[mask],
                ]

        if q_preds and q_tgts:
            q_preds_t = torch.cat(q_preds)
            q_tgts_t = torch.cat(q_tgts)
            return F.mse_loss(q_preds_t, q_tgts_t)
        else:
            return torch.tensor(0.0, device=device)


"""
Observation shape:  ([24, 8, 8, 29]) #[num_env, H,W, C]
move_dir.shape:     ([1, 4, 8, 8])  c=[0,3]  
move_dec.shape:     ([1, 2, 8, 8])  c=[0,1]

"""
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
    """
    ([microrts_ai.passiveAI for _ in range(args.num_bot_envs // 2)] +
          [microrts_ai.workerRushAI for _ in range(args.num_bot_envs // 2)]),
    """
    envs = MicroRTSGridModeVecEnv(
        num_selfplay_envs=args.num_selfplay_envs,
        num_bot_envs=args.num_bot_envs,
        partial_obs=args.partial_obs,
        max_steps=2000,
        render_theme=2,
        ai2s= [microrts_ai.passiveAI for _ in range(args.num_bot_envs)],

        map_paths=[args.train_maps[0]],
        reward_weight=np.array([300.0, 10.0, 40.0, -100.0, 50.0, -100.0]),
        # Win, Ressource, ProduceWorker, Produce Building, Attack, ProduceCombat Unit, (auskommentiert closer to enemy base)
        cycle_maps=args.train_maps,
    )
    envs = MicroRTSStatsRecorder(envs, args.gamma)
    envs = VecMonitor(envs)

    eval_env = MicroRTSGridModeVecEnv(
        num_selfplay_envs=0,
        num_bot_envs=1,
        partial_obs=args.partial_obs,
        max_steps=2000,
        render_theme=2,
        ai2s=[microrts_ai.passiveAI for _ in range(args.num_bot_envs)],
        map_paths=[args.train_maps[0]],
        reward_weight=np.array([300.0, 10.0, 40.0, -100.0, 50.0, -100.0]),
        cycle_maps=args.train_maps
    )
    eval_env = VecMonitor(eval_env)
    #eval_env.seed(args.seed + 999)

    if args.capture_video:
        envs = VecVideoRecorder(
            envs, f"videos/{experiment_name}", record_video_trigger=lambda x: x % 100000 == 0, video_length=2000
        )
    assert isinstance(envs.action_space, MultiDiscrete), "only MultiDiscrete action space is supported"

    eval_executor = None
    if args.max_eval_workers > 0:
        from concurrent.futures import ThreadPoolExecutor

        eval_executor = ThreadPoolExecutor(max_workers=args.max_eval_workers, thread_name_prefix="league-eval-")


    """
    Initialisierung
    """
    expbuffer = ExperienceBuffer(capacity=10000*100)
    agent = Agent(envs, expbuffer, device=device)
    #decset_decision_heads_to_prefer_one(agent)
    total_params = sum(p.numel() for head in agent.heads.values() for p in head.parameters() if p.requires_grad)
    print(f"Gesamtanzahl der trainierbaren Parameter: {total_params}")

    target_heads = {name: head for name, head in agent.heads.items()}

    optimizer = optim.Adam(
        [p for head in agent.heads.values() for p in head.parameters()],
        lr=args.learning_rate
    )

    total_rewards = []
    frame_idx = 0
    episode_idx = 0
    best_mean_reward = 0.0
    epsilon = args.epsilon_start
    mean_reward = 0.0
    reward_queue = deque(maxlen=100)
    warmup_frames=100000
    eval_interval=100000

    if not args.exp_name:
        args.exp_name = "default_exp"
    model_dir=f"./{args.exp_name}/model/"
    os.makedirs(model_dir, exist_ok=True)
    log_dir=f"./{args.exp_name}/csv/"
    os.makedirs(log_dir, exist_ok=True)
    reward_names = [
        "WinLossReward",
        "ResourceGatherReward",
        "ProduceWorkerReward",
        "ProduceBuildingReward",
        "AttackReward",
        "ProduceCombatUnitReward"
    ]
    reward_counts = {name: 0 for name in reward_names}
    frame_start=0
    csv_dir = "./csv"
    os.makedirs(csv_dir, exist_ok=True)
    csv_path=f"./csv/{args.exp_name}.csv"
    print(csv_path)
    print("Starte Training")

    start_time = time.time()
    sync_target_heads(agent.heads, target_heads)  # Direkt synchronisieren
    for name, head in agent.heads.items():
        torch.save(head.state_dict(), os.path.join(model_dir, f"{args.exp_name}_{name}_initial.pth"))

    """
    Training
    """

    while frame_idx < args.total_timesteps:
        frame_idx += 1
        epsilon = max(args.epsilon_final, args.epsilon_start - frame_idx / args.epsilon_decay) #linear
        if frame_idx < warmup_frames:
            epsilon = 1.0
        eval_reward=0.0
        if frame_idx % eval_interval == 0:
            eval_reward = evaluate(agent, eval_env, device=device)
            print(f"[EVAL] Frame {frame_idx} Durchschnittlicher Reward: {eval_reward:.2f}")


        step_info = agent.play_step(epsilon=epsilon, device=device)
        done = step_info["done"]
        reward = step_info["reward"]
        infos = step_info["infos"]

        #print(f"Step: {frame_idx} Done: {done} Reward: {reward}")

        raw_rewards = infos.get("raw_rewards", None)

        for name, value in zip(reward_names, raw_rewards):
                reward_counts[name] += value
        # envs.venv.venv.render(mode="human)

        if len(expbuffer) < args.batch_size:
            continue

        if frame_idx % args.sync_interval == 0:
            sync_target_heads(agent.heads, target_heads)

        if frame_idx % 300000 == 0:
            for name, head in agent.heads.items():
                torch.save(head.state_dict(), f"checkpoints/{args.exp_name}_{name}_{frame_idx}.pth")

        batch = expbuffer.sample(args.batch_size)
        optimizer.zero_grad()
        loss = agent.calc_loss(batch, target_heads, gamma=args.gamma)
        loss.backward()
        optimizer.step()

        # Logging
        if done:
            episode_idx += 1
            reward_queue.append(reward)  # neuen Reward in Queue einfügen
            mean_reward = np.mean(reward_queue)  # Durchschnitt berechnen
            frame_ende=frame_idx
            dauer=frame_ende-frame_start
            frame_start=frame_idx
            #print(envs.ai2s)
            print(f"Episode: {episode_idx} Frame: {frame_idx} "
                  f"Reward: {reward:.2f} Mean Reward: {mean_reward:.2f}  Eval Reward: {eval_reward:.2f} "
                  f"Loss: {loss:.4f} Epsilon: {epsilon} Dauer: {dauer}")
            raw_rewards = infos.get("raw_rewards", None)
            loss_val = loss.item() if loss is not None else None
            log_episode_to_csv(
                csv_path=csv_path,
                episode_idx=episode_idx,
                frame_idx=frame_idx,
                reward=reward,
                mean_reward =mean_reward,
                eval_reward=eval_reward,
                loss=loss_val,
                epsilon=epsilon,
                dauer=dauer,
                reward_counts=reward_counts,
                reward_names=reward_names
            )
            for name, value in zip(reward_names, raw_rewards):
                print(f"{name}: {reward_counts[name]}")

                reward_counts[name]=0
            if frame_idx > warmup_frames and (best_mean_reward is None or mean_reward > best_mean_reward):
                print(f"Neues bestes Ergebnis: old mean reward: {best_mean_reward:.2f} new best mean reward: {mean_reward:.2f}")
                best_mean_reward = mean_reward
                for name, head in agent.heads.items():
                    torch.save(head.state_dict(), os.path.join(model_dir, f"{args.exp_name}_{name}_best.pth"))






    for name, head in agent.heads.items():
        torch.save(head.state_dict(), os.path.join(model_dir, f"{args.exp_name}_{name}_ende.pth"))
    print("Training abgeschlossen.")

    envs.close()
