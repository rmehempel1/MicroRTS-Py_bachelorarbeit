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

class UASDQN(nn.Module):
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
        self.out = nn.Linear(512,95)

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


class ReplayBuffer:
    def __init__(self, capacity, state_shape, action_shape):
        self.buffer = deque(maxlen=capacity)
        self.state_shape = state_shape
        self.action_shape = action_shape

    def append(self, state, action, reward, done, next_state, unit_pos, action_masks, next_action_masks):
        self.buffer.append((state, action, reward, done, next_state, unit_pos, action_masks, next_action_masks ))

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        samples = [self.buffer[idx] for idx in indices]

        states, actions, rewards, dones, next_states, unit_positions, action_masks, next_action_masks = zip(*samples)

        states = np.array(states, copy=False)  # [B, H, W, C]
        actions = np.stack(actions)  # [B, 7]
        """
        print("actions type:", type(actions))
        if isinstance(actions, np.ndarray):
            print("actions shape:", actions.shape)
        else:
            print("actions len:", len(actions), "sample[0] shape:", np.array(actions[0]).shape)
        """
        rewards = np.array(rewards, copy=False)  # [B]
        dones = np.array(dones, copy=False)  # [B]
        next_states = np.array(next_states, copy=False)  # [B, H, W, C]
        unit_positions = np.array(unit_positions, copy=False)  # [B, 2]

        return states, actions, rewards, dones, next_states, unit_positions, action_masks, next_action_masks
class Agent:
    def __init__(self, env, exp_buffer, net, device="cpu"):
        self.env=env
        self.device=device
        self.exp_buffer = exp_buffer
        self._reset()
        self.net = net
        self.state = self.env.reset()
        self.total_rewards = [0.0 for _ in range(self.env.num_envs)]
        self.env_episode_counter = [0 for _ in range(self.env.num_envs)]
        self.total_rewards = [0.0 for _ in range(self.env.num_envs)]
        self.episode_steps = [0 for _ in range(self.env.num_envs)]

    def _reset(self):
        """
        Startet eine neue Episode und setzt interne Zustände zurück.
        """
        self.state = self.env.reset()
        self.total_reward = 0.0

    def qval_to_action(self, q_val: int) -> dict:
        """
        Wandelt einen diskreten Q-Wert (0–88) in eine strukturierte Einzelaktion um.

        Gibt ein Dictionary mit den Feldern:
        - action_type: 1=move, 2=harvest, 3=return, 4=produce, 5=attack
        - move_direction, harvest_direction, return_direction: 0=north, 1=east, 2=south, 3=west
        - produce_type: 0=ressource, ..., 6=ranged
        - produce_direction: 0–3
        - attack_index: 0–48 (relative Zielposition im 7x7-Angriffsraster)
        """
        if not (0 <= q_val <= 88):
            raise ValueError("q_val muss im Bereich 0–88 liegen.")

        action = {
            "action_type": 0,
            "move_direction": 0,
            "harvest_direction": 0,
            "return_direction": 0,
            "produce_direction": 0,
            "produce_type": 0,
            "attack_index": 0,
        }

        if 0 <= q_val < 4:
            action["action_type"] = 1  # move
            action["move_direction"] = q_val

        elif 4 <= q_val < 8:
            action["action_type"] = 2  # harvest
            action["harvest_direction"] = q_val - 4

        elif 8 <= q_val < 12:
            action["action_type"] = 3  # return
            action["return_direction"] = q_val - 8

        elif 12 <= q_val < 40:
            action["action_type"] = 4  # produce
            produce_idx = q_val - 12
            action["produce_type"] = produce_idx // 4
            action["produce_direction"] = produce_idx % 4

        elif 40 <= q_val <= 88:
            action["action_type"] = 5  # attack
            attack_idx = q_val - 40
            action["attack_index"] = attack_idx

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

    def convert_78_to_95_mask(self,mask_78):
        """Konvertiert 78-dim Aktionmaske → 95-diskret"""
        assert mask_78.shape[0] == 78
        m95 = torch.zeros(95, dtype=torch.bool, device=mask_78.device)

        # 0–17 direkt übernehmen
        m95[0:18] = mask_78[0:18]

        # 18–45: produce type × direction (28 Kombinationen)
        for t in range(7):
            if mask_78[22 + t]:  # Typ erlaubt
                for d in range(4):
                    if mask_78[18 + d]:  # Richtung erlaubt
                        idx = 18 + t * 4 + d
                        m95[idx] = True

        # 46–94: Attack (49 Stück)
        m95[46:95] = mask_78[29:78]
        return m95

    def play_step(self, epsilon=0.0):
        net = self.net
        device = self.device

        raw_masks_np = self.env.venv.venv.get_action_mask()  # [num_envs, H*W, 78]
        raw_masks = torch.from_numpy(raw_masks_np).to(device=device).bool()
        _, h, w, _ = self.state.shape
        num_envs = self.env.num_envs
        mask = torch.zeros((num_envs, h, w, 95), dtype=torch.bool, device=device)

        full_action = np.zeros((num_envs, h, w, 7), dtype=np.int32)
        state_v = torch.tensor(self.state.transpose(0, 3, 1, 2), dtype=torch.float32, device=device)

        def sample_valid(mask):
            idx = torch.where(mask)[0]
            return idx[torch.randint(len(idx), (1,))] if len(idx) > 0 else torch.tensor(0, device=device)

        for env_i in range(num_envs):
            for i in range(h):
                for j in range(w):
                    if self.state[env_i, i, j, 11] == 1 and self.state[env_i, i, j, 21] == 1:
                        flat_idx = i * h + j
                        cell_mask_78 = raw_masks[env_i, flat_idx]
                        cell_mask = self.convert_78_to_95_mask(cell_mask_78)

                        mask[env_i, i, j] = cell_mask
                        #print(i,j, cell_mask_78, cell_mask, mask[env_i,i,j])
                        # --- Aktionsauswahl ---
                        if np.random.random() < epsilon:
                            a_type = sample_valid(cell_mask[0:6]).item()
                            full_action[env_i, i, j, 0] = a_type

                            if a_type == 1:
                                full_action[env_i, i, j, 1] = sample_valid(cell_mask[6:10]).item()
                            elif a_type == 2:
                                full_action[env_i, i, j, 2] = sample_valid(cell_mask[10:14]).item()
                            elif a_type == 3:
                                full_action[env_i, i, j, 3] = sample_valid(cell_mask[14:18]).item()
                            elif a_type == 4:
                                prod_idx = sample_valid(cell_mask[18:46]).item()
                                full_action[env_i, i, j, 4] = prod_idx % 4
                                full_action[env_i, i, j, 5] = prod_idx // 4
                            elif a_type == 5:
                                full_action[env_i, i, j, 6] = sample_valid(cell_mask[46:95]).item()
                            print(full_action[env_i,i,j])
                        else:
                            unit_pos = torch.tensor([[j, i]], dtype=torch.float32, device=device)
                            q_vals_v = net(state_v, unit_pos=unit_pos)
                            print(cell_mask)
                            masked_q_vals = q_vals_v.masked_fill(~cell_mask, -1e9)
                            q_val = torch.argmax(masked_q_vals).item()
                            print(q_vals_v, masked_q_vals, q_val)

                            single_action = self.qval_to_action(q_val)
                            print(single_action)
                            full_action[env_i, i, j, 0] = single_action["action_type"]
                            full_action[env_i, i, j, 1] = single_action["move_direction"]
                            full_action[env_i, i, j, 2] = single_action["harvest_direction"]
                            full_action[env_i, i, j, 3] = single_action["return_direction"]
                            full_action[env_i, i, j, 4] = single_action["produce_direction"]
                            full_action[env_i, i, j, 5] = single_action["produce_type"]
                            full_action[env_i, i, j, 6] = single_action["attack_index"]
                            print("Agent Aktion:",full_action[env_i,i,j])

        # --- Schritt ausführen ---
        new_state, reward, is_done, infos = self.env.step(full_action.reshape(1, -1))
        #envs.venv.venv.render(mode="human")
        next_raw_masks = self.env.venv.venv.get_action_mask()

        # --- Replay Buffer befüllen ---
        for env_i in range(num_envs):
            for i in range(h):
                for j in range(w):
                    if self.state[env_i, i, j, 11] == 1 and self.state[env_i, i, j, 21] == 1:
                        flat_idx = i * h + j
                        action_mask = self.convert_78_to_95_mask(raw_masks[env_i, flat_idx]).cpu().numpy()
                        next_action_mask = self.convert_78_to_95_mask(
                            torch.from_numpy(next_raw_masks[env_i, flat_idx]).to(device)
                        ).cpu().numpy()

                        single_action = np.array(full_action[env_i, i, j], dtype=np.int64)

                        self.exp_buffer.append(
                            self.state[env_i],
                            single_action,
                            reward[env_i],
                            is_done[env_i],
                            new_state[env_i],
                            (j, i),
                            action_mask,
                            next_action_mask
                        )

        # --- Episodenabschluss ---
        self.state = new_state
        for env_i in range(num_envs):
            self.total_rewards[env_i] += reward[env_i]
            self.episode_steps[env_i] += 1

            if is_done[env_i]:
                ep = self.env_episode_counter[env_i]
                shaped = self.total_rewards[env_i]
                raw = infos[env_i].get("raw_rewards", None)
                steps = self.episode_steps[env_i]

                print(f"[Env {env_i} | Episode {ep}] Reward: {shaped:.2f}, RawReward: {raw}, Steps: {steps}")

                self.env_episode_counter[env_i] += 1
                self.total_rewards[env_i] = 0.0
                self.episode_steps[env_i] = 0

        return {"done": False}

    def calc_loss(self, batch, tgt_net, gamma):
        import torch.nn.functional as F

        states, actions, rewards, dones, next_states, unit_positions, action_masks, next_action_masks = batch
        device = self.device
        B = len(actions)

        # --- Tensor-Vorbereitung ---
        states_v = torch.tensor(states, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
        next_states_v = torch.tensor(next_states, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
        actions_v = torch.tensor(actions, dtype=torch.int64, device=device)  # [B, 7]
        rewards_v = torch.tensor(rewards, dtype=torch.float32, device=device)
        done_mask = torch.tensor(dones, dtype=torch.bool, device=device)
        unit_pos = torch.tensor(unit_positions, dtype=torch.long, device=device)

        # --- Q(s, a) und Q(s', ·) berechnen ---
        qvals = self.net(states_v, unit_pos=unit_pos)  # [B, 95]
        qvals_next = tgt_net(next_states_v, unit_pos=unit_pos)  # [B, 95]

        # --- Aktionsindex bestimmen ---
        q_indices = torch.tensor(
            [self.action_to_qval(a.tolist()) for a in actions_v], device=device
        )
        state_action_qvals = qvals[torch.arange(B), q_indices]  # Q(s,a)

        # --- TD-Ziel berechnen ---
        target_qvals = torch.zeros(B, dtype=torch.float32, device=device)
        for b in range(B):
            if next_action_masks is not None:
                # Umwandlung zurück zu Tensor
                next_mask_95 = torch.tensor(next_action_masks[b], dtype=torch.bool, device=device)
                if next_mask_95.any():
                    max_q = qvals_next[b][next_mask_95].max()
                else:
                    max_q = torch.tensor(0.0, device=device)
            else:
                max_q = qvals_next[b].max()

            target_qvals[b] = rewards_v[b] if done_mask[b] else rewards_v[b] + gamma * max_q

        loss = F.smooth_l1_loss(state_action_qvals, target_qvals)
        return loss


def log_episode_to_csv(
    csv_path: str,
    episode_idx: int,
    frame_idx: int,
    reward: float,
    mean_reward: float,
    eval_reward: float,
    losses: list,
    epsilon: float,
    dauer: float,
    reward_counts: dict,
    reward_names: list
):
    """
    Schreibt eine abgeschlossene Episode in eine CSV-Datei mit robustem Logging.
    Konvertiert alle Tensoren in lesbare Floats. Fügt optional Diagnose-Informationen hinzu.
    """
    def to_float(val):
        # Konvertiert Tensoren oder einfache Zahlen in float
        if isinstance(val, torch.Tensor):
            return val.item()
        elif isinstance(val, (float, int)):
            return float(val)
        else:
            try:
                return float(val)
            except:
                return str(val)

    # Spaltenüberschriften
    header = ["episode", "frame", "reward", "mean_reward", "eval_reward"]
    header += [f"loss_head_{i}" for i in range(len(losses))]
    header += ["epsilon", "dauer"] + reward_names

    # Inhalte vorbereiten
    row = [
        episode_idx,
        frame_idx,
        reward,
        mean_reward,
        eval_reward,
    ] + [to_float(l) for l in losses] + [epsilon, dauer] + [reward_counts.get(name, 0) for name in reward_names]

    # Datei öffnen und ggf. Header schreiben
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists or os.stat(csv_path).st_size == 0:
            writer.writerow(header)
        writer.writerow(row)

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
    print("PID: ", os.getpid())
    print(f"Device: {device}")
    #d
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    """
    ([microrts_ai.passiveAI for _ in range(args.num_bot_envs // 2)] +
          [microrts_ai.workerRushAI for _ in range(args.num_bot_envs // 2)]),
    """
    reward_weights = np.array([10.0, 3.0, 3.0, 0.0, 5.0, 1.0])
    print("Reward Weights:", reward_weights)
    num_envs = 1#args.num_bot_envs
    num_each = 1#num_envs // 2  # ganzzahliger Anteil
    envs = MicroRTSGridModeVecEnv(
        num_selfplay_envs=args.num_selfplay_envs,
        num_bot_envs=args.num_bot_envs,
        partial_obs=args.partial_obs,
        max_steps=args.max_steps,
        render_theme=2,
        ai2s=  (
    [microrts_ai.workerRushAI for _ in range(num_each)] +
    [microrts_ai.passiveAI for _ in range(num_envs - num_each)]
),

        map_paths=[args.train_maps[0]],
        reward_weight=reward_weights,
        # Win, Ressource, ProduceWorker, Produce Building, Attack, ProduceCombat Unit, (auskommentiert closer to enemy base)
        cycle_maps=args.train_maps,
    )
    envs = MicroRTSStatsRecorder(envs, args.gamma)
    envs = VecMonitor(envs)
    """
    eval_env = MicroRTSGridModeVecEnv(
        num_selfplay_envs=0,
        num_bot_envs=1,
        partial_obs=args.partial_obs,
        max_steps=args.max_steps,
        render_theme=2,
        ai2s=[microrts_ai.passiveAI for _ in range(args.num_bot_envs)],
        map_paths=[args.train_maps[0]],
        reward_weight=reward_weights,
        cycle_maps=args.train_maps
    )
    eval_env = VecMonitor(eval_env)
    #eval_env.seed(args.seed + 999)
    """
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
    dummy_obs = envs.reset()
    state_shape = dummy_obs.shape[1:]  # [H, W, C]
    action_shape = (7,)  # [H, W, 7] später

    expbuffer = ReplayBuffer(capacity=args.buffer_memory, state_shape=state_shape, action_shape=action_shape)

    dummy_input_shape = (29, 8, 8)  # [C, H, W]
    policy_net = UASDQN(input_shape=dummy_input_shape).to(device)
    target_net = UASDQN(input_shape=dummy_input_shape).to(device)
    target_net.load_state_dict(policy_net.state_dict())  #  Initiales Sync
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.learning_rate)

    # Agent vorbereiten
    agent = Agent(env=envs, exp_buffer=expbuffer, net=policy_net, device=device)

    # Parameterzähler
    total_params = sum(p.numel() for p in policy_net.parameters() if p.requires_grad)
    print(f"Gesamtanzahl der trainierbaren Parameter: {total_params}")

    # Weitere Initialisierung
    frame_idx = 0
    episode_idx = 0
    best_mean_reward = None
    reward_queue = deque(maxlen=100)
    warmup_frames = args.warmup_frames
    eval_interval = 100000
    frame_start = 0

    # Ordner
    model_dir = f"./{args.exp_name}/model/"
    os.makedirs(model_dir, exist_ok=True)
    csv_path = f"./csv/{args.exp_name}.csv"
    os.makedirs("./csv", exist_ok=True)

    # Reward-Tracking
    reward_names = [
        "WinLossReward", "ResourceGatherReward", "ProduceWorkerReward",
        "ProduceBuildingReward", "AttackReward", "ProduceCombatUnitReward"
    ]
    reward_counts = {name: 0 for name in reward_names}

    print("Starte Training")
    print("Learning Rate:", args.learning_rate)


    """
    Training
    """
    # Netzwerke initialisieren
    target_net.load_state_dict(policy_net.state_dict())  #  Initiales Sync
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.learning_rate)

    # Agent vorbereiten
    agent = Agent(env=envs, exp_buffer=expbuffer, net=policy_net, device=device)

    print(f"Total trainierbare Parameter: {sum(p.numel() for p in policy_net.parameters() if p.requires_grad)}")

    # Training
    while frame_idx < args.total_timesteps:
        frame_idx += 1
        epsilon = max(args.epsilon_final, args.epsilon_start - frame_idx / args.epsilon_decay)
        if frame_idx < warmup_frames:
            epsilon = 0.0

        eval_reward = 0.0
        #if frame_idx % eval_interval == 0:
            #eval_reward = evaluate(agent, eval_env, device=device)
            #print(f"[EVAL] Frame {frame_idx} Durchschnittlicher Reward: {eval_reward:.2f}")

        # Schritt ausführen
        #print("frame:" ,frame_idx)
        step_info = agent.play_step(epsilon=epsilon)
        #envs.venv.venv.render(mode="human")
        #done = step_info["done"]
        #print("done: ", done)
        #reward = step_info["reward"]
        #infos = step_info["infos"]
        #raw_rewards = sum(info.get("raw_rewards", 0.0) for info in infos)

        """        for name, value in zip(reward_names, raw_rewards):
            reward_counts[name] += value"""

        if len(expbuffer) < args.batch_size:
            #print("buffer: ", len(expbuffer))
            continue
        #print("buffer:", len(expbuffer))

        # Target-Sync
        if frame_idx % args.sync_interval == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Checkpoint speichern
        if frame_idx % 300000 == 0:
            torch.save(policy_net.state_dict(), f"checkpoints/{args.exp_name}_{frame_idx}.pth")

        # Training
        batch = expbuffer.sample(args.batch_size)
        optimizer.zero_grad()

        loss = agent.calc_loss(batch, target_net, gamma=args.gamma)
        loss.backward()
        optimizer.step()

        # Logging



    # Training fertig – final speichern
    torch.save(policy_net.state_dict(), os.path.join(model_dir, f"{args.exp_name}_final.pth"))
    print("Training abgeschlossen.")

    envs.close()



