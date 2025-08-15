import argparse
import os
import random
import csv
import time
from distutils.util import strtobool
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

class UASDRQN(nn.Module):
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

        self.rnn = nn.GRU(input_size=512, hidden_size=512, batch_first=True)
        """
        4 move
        4 harvest
        4 return
        4*7 produce direction
        49 attack dir"""
        self.out = nn.Linear(512,89)

    def forward(self, x, unit_pos, hidden=None, return_sequence=False):
        """
        x:        [B, T, C, H, W] - Zustände als Sequenz
        unit_pos: [B, 2]          - aktuelle (x, y)-Position im letzten Frame
        hidden:   optionaler initialer Hidden State für RNN [1, B, hidden_dim]
        return_sequence: wenn True, gibt gesamte Sequenz zurück, sonst nur letzten Schritt
        """
        B, T, C, H, W = x.size()

        # --- CNN-Feature-Extraktion ---
        x = x.view(B * T, C, H, W)  # [B*T, C, H, W]
        feat = self.encoder(x)  # [B*T, F]
        feat = feat.reshape(B, T, -1)  # [B, T, F]

        # --- Positional Encoding nur für letzten Frame ---
        norm_x = unit_pos[:, 0].float() / (self.input_width - 1)
        norm_y = unit_pos[:, 1].float() / (self.input_height - 1)
        pos_enc_last = torch.stack([norm_x, norm_y], dim=1)  # [B, 2]

        # Die Positional-Encoding-Info nur an letzten Step anhängen
        # Vorher alle Steps mit 0-Posen auffüllen
        pos_enc_seq = torch.zeros(B, T, 2, device=feat.device)
        pos_enc_seq[:, -1, :] = pos_enc_last  # nur letzter Step hat echte Position

        # Features + Position
        feat_with_pos = torch.cat([feat, pos_enc_seq], dim=2)  # [B, T, F+2]

        # --- Vollverbundene Schicht vor RNN ---
        feat_with_pos = self.fc(feat_with_pos)  # [B, T, hidden_dim]

        # --- RNN über gesamte Sequenz ---
        rnn_out, hidden_out = self.rnn(feat_with_pos, hidden)  # [B, T, hidden_dim]

        if return_sequence:
            out = self.out(rnn_out)  # [B, T, action_dim]
        else:
            out = self.out(rnn_out[:, -1, :])  # [B, action_dim]

        return out, hidden_out

    def _get_conv_out(self, shape):
        o = self.encoder(torch.zeros(1, *shape))
        return int(np.prod(o.size()))


class ReplayBuffer:
    def __init__(self, capacity, state_shape, action_shape, seq_len=1):
        self.buffer = deque(maxlen=capacity)
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.seq_len = seq_len

    def _to_numpy(self, x):
        """Konvertiert Tensoren (CUDA/CPU) oder Listen zu NumPy."""
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        elif isinstance(x, (list, tuple)):
            # Rekursiv auf alle Elemente anwenden und stacken
            return np.array([self._to_numpy(el) for el in x], copy=False)
        else:
            return np.array(x, copy=False)

    def append(self, state, action, reward, done, next_state, unit_pos, action_masks, next_action_masks):
        self.buffer.append((
            self._to_numpy(state),
            self._to_numpy(action),
            self._to_numpy(reward),
            self._to_numpy(done),
            self._to_numpy(next_state),
            self._to_numpy(unit_pos),
            self._to_numpy(action_masks),
            self._to_numpy(next_action_masks)
        ))

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        samples = [self.buffer[idx] for idx in indices]

        states, actions, rewards, dones, next_states, unit_positions, action_masks, next_action_masks = zip(*samples)

        # Direkt NumPy-Arrays zurückgeben
        return (
            np.array(states, copy=False),
            np.array(actions, copy=False),
            np.array(rewards, copy=False),
            np.array(dones, copy=False),
            np.array(next_states, copy=False),
            np.array(unit_positions, copy=False),
            np.array(action_masks, copy=False),
            np.array(next_action_masks, copy=False)
        )

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

    def play_step(self, epsilon=0.0):
        net = self.net
        device = self.device
        episode_results = []

        num_envs = self.env.num_envs
        _, h, w, _ = self.state.shape
        seq_len = getattr(self.exp_buffer, "seq_len", 8)

        # --- Historienpuffer initialisieren ---
        if not hasattr(self, "seq_buffers"):
            # pro Env: Liste von States (nur s_t, ohne Actions etc.)
            self.seq_buffers = [[] for _ in range(num_envs)]
        if not hasattr(self, "unit_seq_buffers"):
            # pro Unit (env,i,j): deque mit Transitions
            from collections import defaultdict, deque
            self.unit_seq_buffers = defaultdict(lambda: deque(maxlen=seq_len * 2))

        # 1) Aktuellen State s_t in die Entscheidungs-Historie legen (für RNN-Input)
        for env_i in range(num_envs):
            self.seq_buffers[env_i].append((self.state[env_i].copy(),))  # nur State an Pos 0

        # 2) Aktionsmasken + aktive Einheiten bestimmen
        raw_masks_np = self.env.venv.venv.get_action_mask()  # [E, H*W, 78]
        raw_masks = torch.from_numpy(raw_masks_np).to(device=device).bool()
        units_mask_np = (self.state[..., 11] == 1) & (self.state[..., 21] == 1)  # [E,H,W]
        units_mask = torch.from_numpy(units_mask_np).to(device=device)
        env_idx, i_idx, j_idx = torch.where(units_mask)
        K = env_idx.numel()

        full_action = np.zeros((num_envs, h, w, 7), dtype=np.int32)

        # 3) Aktion wählen (ε-greedy) – Sequenz [T,C,H,W] pro Unit ins Netz
        if K > 0:
            flat_idx = i_idx * w + j_idx
            cell_masks_89 = torch.stack([
                self.convert_78_to_89_mask(raw_masks[e.item(), f.item()])
                for e, f in zip(env_idx, flat_idx)
            ], dim=0).to(device=device)  # [K,89] bool

            for n in range(K):
                e = int(env_idx[n].item())
                ii = int(i_idx[n].item())
                jj = int(j_idx[n].item())

                # --- Sequenz aus seq_buffers[e] holen & ggf. pad ---
                seq = self.seq_buffers[e][-seq_len:]  # Elemente sind Tuples: (state,)
                if len(seq) == 0:
                    state_seq = [self.state[e]] * seq_len
                else:
                    state_seq = [frame[0] for frame in seq]  # nur States ziehen
                    if len(state_seq) < seq_len:
                        state_seq = [state_seq[0]] * (seq_len - len(state_seq)) + state_seq

                # [T,H,W,C] -> [T,C,H,W]
                state_seq_t = torch.tensor(np.array(state_seq), dtype=torch.float32, device=device).permute(0, 3, 1, 2)

                # Positionsencoding (nur letzter Step)
                unit_pos_t = torch.tensor([jj, ii], dtype=torch.float32, device=device).unsqueeze(0)  # [1,2]

                # Netzaufruf: [B=1,T,C,H,W] -> Q[last]
                q_vals, _ = net(state_seq_t.unsqueeze(0), unit_pos=unit_pos_t)  # [1,89]
                q_vals = q_vals.squeeze(0)  # [89]

                # Maskieren + ε-greedy
                mask_89 = cell_masks_89[n]
                if mask_89.sum() == 0:
                    action_idx = 0  # Fallback
                else:
                    masked_q = q_vals.masked_fill(~mask_89, -1e9)
                    if torch.rand(1, device=device) < epsilon:
                        valid = torch.nonzero(mask_89, as_tuple=False).squeeze(1)
                        action_idx = valid[torch.randint(len(valid), (1,), device=device)].item()
                    else:
                        action_idx = masked_q.argmax().item()

                full_action[e, ii, jj] = self.qval_to_action(action_idx)

        # 4) Schritt ausführen
        prev_state = self.state.copy()
        new_state, reward, is_done, infos = self.env.step(full_action.reshape(num_envs, -1))
        next_raw_masks = self.env.venv.venv.get_action_mask()  # [E, H*W, 78]
        self.state = new_state  # s_{t+1}

        # 5) Transitions pro Unit in ihren eigenen Sequenzpuffer legen + ggf. Replay-Append
        for n in range(K):
            e = int(env_idx[n].item())
            ii = int(i_idx[n].item())
            jj = int(j_idx[n].item())
            flat = ii * w + jj

            # Aktion als Q-Index
            single_action_arr = np.array(full_action[e, ii, jj], dtype=np.int64)
            a_qidx = self.action_to_qval(single_action_arr)

            # Masken (s_t, s_{t+1})
            act_mask = self.convert_78_to_89_mask(raw_masks[e, flat])
            next_act_mask = self.convert_78_to_89_mask(next_raw_masks[e, flat])

            key = (e, ii, jj)
            self.unit_seq_buffers[key].append((
                prev_state[e],  # s_t [H,W,C]
                a_qidx,  # a_t  (Q-index)
                reward[e],  # r_{t+1}
                is_done[e],  # done
                new_state[e],  # s_{t+1}
                (jj, ii),  # (x=j, y=i) für letzten Step
                act_mask,  # gültige in s_t
                next_act_mask  # gültige in s_{t+1}
            ))

            if len(self.unit_seq_buffers[key]) >= seq_len:
                seq = list(self.unit_seq_buffers[key])[-seq_len:]
                (states, actions, rewards_, dones, next_states,
                 unit_positions, action_masks, next_action_masks) = zip(*seq)

                self.exp_buffer.append(states, actions, rewards_, dones,
                                       next_states, unit_positions,
                                       action_masks, next_action_masks)

        # 6) Episode-Ende / Logging / Aufräumen
        for env_i in range(num_envs):
            self.total_rewards[env_i] += reward[env_i]
            self.episode_steps[env_i] += 1
            if is_done[env_i]:
                ep = self.env_episode_counter[env_i]
                shaped = self.total_rewards[env_i]
                steps = self.episode_steps[env_i]
                raw_stats = infos[env_i].get("microrts_stats", {})
                episode_info = {
                    "env": env_i, "episode": ep,
                    "reward": shaped, "steps": steps, "epsilon": epsilon
                }
                for k in ["WinLoss", "ResourceGather", "ProduceWorker", "ProduceBuilding", "Attack",
                          "ProduceCombatUnit"]:
                    episode_info[k] = raw_stats.get(f"{k}RewardFunction", 0.0)
                episode_results.append(episode_info)

                # Reset Zähler
                self.env_episode_counter[env_i] += 1
                self.total_rewards[env_i] = 0.0
                self.episode_steps[env_i] = 0

                # Verlaufs- und Unit-Puffer dieser Env löschen (keine Episodenvermischung)
                self.seq_buffers[env_i].clear()
                to_del = [k for k in list(self.unit_seq_buffers.keys()) if k[0] == env_i]
                for k in to_del:
                    del self.unit_seq_buffers[k]

        return {"done": False, "episode_stats": episode_results}

    def calc_loss(self, batch, tgt_net, gamma):
        states, actions, rewards, dones, next_states, unit_positions, action_masks, next_action_masks = batch
        device = self.device
        B, T = actions.shape[:2]

        # --- Tensor-Vorbereitung ---
        # States: [B, T, C, H, W]
        states_v = torch.tensor(states, dtype=torch.float32, device=device).permute(0, 1, 4, 2, 3)
        next_states_v = torch.tensor(next_states, dtype=torch.float32, device=device).permute(0, 1, 4, 2, 3)
        actions_v = torch.tensor(actions, dtype=torch.int64, device=device)  # [B, T]
        rewards_v = torch.tensor(rewards, dtype=torch.float32, device=device)  # [B, T]
        done_mask = torch.tensor(dones, dtype=torch.bool, device=device)  # [B, T]
        unit_pos_v = torch.tensor(unit_positions[:, -1, :], dtype=torch.long, device=device)  # [B, 2] nur letzter Step

        # --- Q(s, a) für aktuelle Sequenz ---
        qvals, _ = self.net(states_v, unit_pos=unit_pos_v)  # qvals: [B, action_dim] nur letzter Step
        # -> Wir gehen davon aus, dass dein Netz so gebaut ist, dass es bei Sequenzen den letzten Step ausgibt

        # --- Q(s', ·) für nächste Sequenz ---
        #  hier einfach nur die Unit-Position aus dem letzten Schritt der Next-State-Sequenz.
        next_unit_pos_v = unit_pos_v.clone()  # falls du Positionsencoding nur für aktuellen Frame nutzt
        qvals_next, _ = tgt_net(next_states_v, unit_pos=next_unit_pos_v)  # [B, action_dim]

        # --- Aktionen & Rewards nur für letzten Step ---
        last_actions = actions_v[:, -1]  # [B]
        last_rewards = rewards_v[:, -1]  # [B]
        last_dones = done_mask[:, -1]  # [B]

        # --- Q(s,a) extrahieren ---
        state_action_qvals = qvals[torch.arange(B, device=device), last_actions]  # [B]

        # --- Maskierung für nächste Q-Werte ---
        if next_action_masks is not None:
            last_next_masks = [m[-1] for m in next_action_masks]  # nur letzter Step
            next_masks_v = torch.stack(
                [torch.as_tensor(m, dtype=torch.bool, device=device) for m in last_next_masks]
            )  # [B, action_dim]
            masked_qvals_next = qvals_next.masked_fill(~next_masks_v, float('-inf'))
            max_qvals_next, _ = masked_qvals_next.max(dim=1)
            max_qvals_next[torch.isinf(max_qvals_next)] = 0.0
        else:
            max_qvals_next, _ = qvals_next.max(dim=1)  # [B]

        # --- Zielwerte ---
        target_qvals = torch.where(
            last_dones,
            last_rewards,
            last_rewards + gamma * max_qvals_next
        )

        # --- Loss ---
        loss = F.smooth_l1_loss(state_action_qvals, target_qvals)
        return loss




if __name__ == "__main__":
    print(">>> Argumentparser wird initialisiert")
    args = parse_args()

    # NEU: Sequenzlänge als Parameter (falls nicht schon in parse_args vorhanden)
    if not hasattr(args, "seq_len"):
        args.seq_len = 8  # Defaultwert, kann in CLI überschrieben werden

    print(f"Save frequency: {args.save_frequency}, seq_len: {args.seq_len}")

    experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.prod_mode:
        import wandb
        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
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

    # Seeding
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print("PID: ", os.getpid())
    print(f"Device: {device}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Environment Setup
    reward_weights = np.array([50.0, 3.0, 3.0, 0.0, 5.0, 1.0])
    num_envs = args.num_bot_envs
    num_each = num_envs // 4
    envs = MicroRTSGridModeVecEnv(
        num_selfplay_envs=args.num_selfplay_envs,
        num_bot_envs=args.num_bot_envs,
        partial_obs=args.partial_obs,
        max_steps=args.max_steps,
        render_theme=2,
        ai2s=(
            [microrts_ai.passiveAI for _ in range(num_each)] +
            [microrts_ai.workerRushAI for _ in range(num_each)] +
            [microrts_ai.lightRushAI for _ in range(num_each)] +
            [microrts_ai.coacAI for _ in range(num_envs - 3 * num_each)]
        ),
        map_paths=[args.train_maps[0]],
        reward_weight=reward_weights,
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

    # Initialisierung
    dummy_obs = envs.reset()
    state_shape = dummy_obs.shape[1:]  # [H, W, C]
    action_shape = (7,)  # [H, W, 7] später

    # NEU: ReplayBuffer mit Sequenzen
    expbuffer = ReplayBuffer(
        capacity=args.buffer_memory,
        state_shape=state_shape,
        action_shape=action_shape,
        seq_len=args.seq_len
    )

    # Dummy-Eingabe für Netz (C,H,W)
    dummy_input_shape = (state_shape[2], state_shape[0], state_shape[1])
    policy_net = UASDRQN(input_shape=dummy_input_shape).to(device)
    target_net = UASDRQN(input_shape=dummy_input_shape).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    policy_net.train()
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.learning_rate)

    # Agent mit neuem play_step (Sequenz-fähig)
    agent = Agent(env=envs, exp_buffer=expbuffer, net=policy_net, device=device)

    total_params = sum(p.numel() for p in policy_net.parameters() if p.requires_grad)
    print(f"Gesamtanzahl der trainierbaren Parameter: {total_params}")

    frame_idx = 0
    best_mean_reward = None
    reward_queue = deque(maxlen=100)
    warmup_frames = args.warmup_frames
    eval_interval = 100000

    # Ordner
    model_dir = f"./{args.exp_name}/model/"
    os.makedirs(model_dir, exist_ok=True)
    csv_path = f"./csv/{args.exp_name}.csv"
    os.makedirs("./csv", exist_ok=True)

    reward_names = [
        "WinLossReward", "ResourceGatherReward", "ProduceWorkerReward",
        "ProduceBuildingReward", "AttackReward", "ProduceCombatUnitReward"
    ]
    reward_counts = {name: 0 for name in reward_names}

    print(f"Starte Training [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
    print("Learning Rate:", args.learning_rate)

    target_net.load_state_dict(policy_net.state_dict())

    torch.save(policy_net.state_dict(), f"./{args.exp_name}/{args.exp_name}_initial.pth")

    initial_sd = {k: v.detach().clone() for k, v in policy_net.state_dict().items()}

    # Training
    while frame_idx < args.total_timesteps:
        frame_idx += 1

        if frame_idx % 1000 == 0:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Frame idx: {frame_idx}")

        epsilon = max(args.epsilon_final, args.epsilon_start - frame_idx / args.epsilon_decay)
        if frame_idx < warmup_frames:
            epsilon = 1.0

        log_path = f"./{args.exp_name}/{args.exp_name}_train_log.csv"
        file_exists = os.path.exists(log_path)

        step_info = agent.play_step(epsilon=epsilon)

        for ep_data in step_info.get("episode_stats", []):
            with open(log_path, "a") as f:
                ep_data_with_frame = dict(ep_data)
                ep_data_with_frame["frame_idx"] = frame_idx
                if not file_exists or os.stat(log_path).st_size == 0:
                    header = list(ep_data_with_frame.keys())
                    f.write(",".join(header) + "\n")
                else:
                    header = list(ep_data_with_frame.keys())
                values = [str(ep_data_with_frame[k]) for k in header]
                f.write(",".join(values) + "\n")

        if len(expbuffer) < args.batch_size:
            continue

        # Target-Sync
        if frame_idx % args.sync_interval == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Checkpoint
        if frame_idx % args.save_network == 0:
            save_path = os.path.join(model_dir, f"{args.exp_name}_{frame_idx}.pth")
            torch.save(policy_net.state_dict(), save_path)
            print(f"Checkpoint gespeichert: {save_path}")

        # Training-Schritt
        batch = expbuffer.sample(args.batch_size)
        optimizer.zero_grad()
        loss = agent.calc_loss(batch, target_net, gamma=args.gamma)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            total_diff = 0.0
            for k, v in policy_net.state_dict().items():
                total_diff += (v - initial_sd[k]).abs().sum().item()
            if frame_idx % 10000 == 0:
                print(f"[dbg] Param-Δ L1 seit Start: {total_diff:.4f}")

    # Training fertig – final speichern
    torch.save(policy_net.state_dict(), f"./{args.exp_name}/{args.exp_name}_final.pth")
    print("Training abgeschlossen.")
    envs.close()






