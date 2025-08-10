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
    """
    Deep Recurrent Q-Network für MicroRTS-ähnliche Zustände.

    - CNN-Encoder extrahiert räumliche Features pro Zeitstufe
    - GRU modelliert zeitliche Abhängigkeiten (rekurrente Schicht)
    - Vollverbinder + Linear geben Q-Werte für 89 diskrete Aktionen aus

    Erwartete Eingaben:
    - x: [B, T, C, H, W] für Sequenzen oder [B, C, H, W] für Einzelschritte
    - unit_pos: Optional Positions-Feature (x,y) normalisiert auf [0,1]
        * für Sequenzen: [B, T, 2]
        * für Einzelschritte: [B, 2]
    - h0: optionaler initialer GRU-Hidden-State [1, B, hidden_size]

    Rückgaben:
    - q: Q-Werte
        * Sequenz-Input: [B, T, n_actions]
        * Einzelschritt: [B, n_actions]
    - h_n: letzter Hidden-State der GRU [1, B, hidden_size]

    Hinweise:
    - Für step-by-step Interaktion (z.B. bei Evaluation) kann man pro Schritt
      x in Form [B, C, H, W] übergeben und den h_n in den nächsten Schritt
      einspeisen (online RNN).
    - Für BPTT im Training über Sequenzen x als [B, T, C, H, W] übergeben.
    """

    def __init__(self, input_shape, hidden_size: int = 512, n_actions: int = 89):
        super().__init__()
        c, h, w = input_shape

        self.input_height = h
        self.input_width = w
        self.hidden_size = hidden_size
        self.n_actions = n_actions

        # CNN-Encoder (leicht tiefer als Original für Stabilität)
        self.encoder = nn.Sequential(
            nn.Conv2d(c, c * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c * 2, c * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        conv_out_size = self._get_conv_out((c, h, w))

        # Feature-Projektion (+2 für Positionskanal)
        self.feature = nn.Sequential(
            nn.Linear(conv_out_size + 2, hidden_size),
            nn.ReLU(inplace=True),
        )

        # Rekurrente Schicht
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)

        # Kopf für Q-Werte
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, n_actions),
        )

        self._init_weights()

    def _init_weights(self):
        # Kaiming für Linear/Conv, orthogonal für GRU
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')  # initiert mit kaiming
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.kaiming_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    # ----------------------------
    # Vorwärtsläufe
    # ----------------------------
    def forward(self,x,unit_pos = None,h0 = None):
        """
        Forward für Sequenz oder Einzelschritt.
        """
        is_sequence = (x.dim() == 5)  # [B,T,C,H,W]

        B, T, C, H, W = x.shape

        # CNN über jedes Zeitfenster
        x = x.reshape(B * T, C, H, W)
        feat_map = self.encoder(x)  # [B*T, C', H, W]
        feat_vec = feat_map.reshape(B * T, -1)

        # Positionsfeature vorbereiten

        up = unit_pos.reshape(B * T, 2).float()
        # Normalisierung auf [0,1]
        norm_x = up[:, 0] / (self.input_width - 1)
        norm_y = up[:, 1] / (self.input_height - 1)
        pos = torch.stack([norm_x, norm_y], dim=1)
        pos = pos.to(feat_vec.device)

        feat = torch.cat([feat_vec, pos], dim=1)
        feat = self.feature(feat)  # [B*T, H]
        feat = feat.view(B, T, -1)  # [B, T, H]

        # Rekurrenz
        if h0 is None:
            # Standard: Null-Initialisierung auf richtigem Device
            h0 = feat.new_zeros(1, B, self.hidden_size)
        out_seq, h_n = self.gru(feat, h0)

        # Q-Werte Kopf
        q_seq = self.head(out_seq)  # [B, T, n_actions]

        if not is_sequence:
            # Einzelschritt zurückformen
            q_seq = q_seq.squeeze(1)  # [B, n_actions]

        return q_seq, h_n  # h_n: [1,B,H]

    # Hilfsfunktion zur Ermittlung der Encoder-Outputgröße
    def _get_conv_out(self, shape):
        with torch.no_grad():
            o = self.encoder(torch.zeros(1, *shape))
            return int(np.prod(o.size()))


import numpy as np
from collections import deque

class ReplayBuffer:
    """
    Replay-Buffer mit Unterstützung für:
      - 1-Step (Standard)
      - N-Step-Returns (online)
      - λ-Returns (episodenweise; benötigt value_fn)

    Hinweise:
    - Für DQN ist ein gängiger Value-Schätzer: V(s) = max_a Q_target(s, a | mask)
    - Für λ-Returns werden am Episodenende alle Übergänge mit G^λ berechnet und in den Hauptpuffer geschrieben.
    """
    def __init__(self, capacity, state_shape, action_shape,
                 gamma=0.99,
                 n_step=1,
                 use_lambda=False,
                 lam=0.95,
                 value_fn=None):
        self.buffer = deque(maxlen=capacity)
        self.state_shape = state_shape
        self.action_shape = action_shape

        # Rückgabe-Optionen
        self.gamma = float(gamma)
        self.n_step = int(n_step)
        self.use_lambda = bool(use_lambda)
        self.lam = float(lam)
        self.value_fn = value_fn  # callable(next_state, next_action_mask, unit_pos_next) -> V(next_state)

        # Für N-step Aggregation (online)
        self._nstep_queue = deque()  # enthält (s,a,r,done,s_next,pos,mask,mask_next)
        self._nstep_R = 0.0
        self._nstep_gamma = 1.0

        # Für λ-Returns (Episodenpuffer)
        self._episode = []  # list of transitions wie oben

    def __len__(self):
        return len(self.buffer)

    # ----------- öffentliche API -----------
    def append(self, state, action, reward, done, next_state, unit_pos, action_masks, next_action_masks):
        tr = (state, action, reward, done, next_state, unit_pos, action_masks, next_action_masks)

        if self.use_lambda:
            # Sammle die Episode; schreibe erst am Ende G^λ in self.buffer
            self._episode.append(tr)
            if done:
                self._flush_episode_lambda()
                self._episode.clear()
        else:
            # N-step (inkl. 1-step als Spezialfall)
            self._push_nstep(tr)

    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        samples = [self.buffer[i] for i in idx]
        states, actions, rewards, dones, next_states, unit_positions, action_masks, next_action_masks = zip(*samples)

        states = np.array(states, copy=False)                 # [B, H, W, C]  (deine Ordnung beibehalten)
        actions = np.stack(actions)                           # [B, ...]
        rewards = np.array(rewards, copy=False).astype(np.float32)  # [B]
        dones = np.array(dones, copy=False).astype(np.bool_)         # [B]
        next_states = np.array(next_states, copy=False)       # [B, H, W, C]
        unit_positions = np.array(unit_positions, copy=False) # [B, 2]
        # action_masks / next_action_masks bleiben als Tuple/Liste (evtl. heterogene Shapes)

        return states, actions, rewards, dones, next_states, unit_positions, action_masks, next_action_masks

    # Wenn du das Training vor Episodenende abbrichst (z. B. Timeout), kannst du offene Episoden/Queues flushen:
    def finalize_pending(self):
        if self.use_lambda and len(self._episode) > 0:
            self._flush_episode_lambda()
            self._episode.clear()
        # Restliche n-step Übergänge ohne weiteren Bootstrap (nur falls gewünscht):
        self._flush_nstep_tail()

    # ----------- interne Helfer -----------
    def _push_nstep(self, tr):
        """Online N-step Aggregation. Schreibt fertige N-step-Übergänge in self.buffer."""
        s, a, r, done, s_next, pos, mask, mask_next = tr
        self._nstep_queue.append(tr)

        # Akkumuliere Return und Discount
        self._nstep_R += self._nstep_gamma * float(r)
        self._nstep_gamma *= self.gamma

        # Sobald wir n Schritte haben oder die Episode terminiert, schreiben wir den Übergang
        if len(self._nstep_queue) >= self.n_step or done:
            # Head-Transition (s_t, a_t, ..., mask_t)
            s0, a0, _, d0, _, pos0, mask0, _ = self._nstep_queue[0]

            # N-step Bootstrap-Zustand ist der letzte der Queue
            _, _, _, d_last, s_last, pos_last, _, mask_last = self._nstep_queue[-1]

            # Wenn die Episode vorzeitig endet, bootstrapen wir nicht über den Terminus hinaus
            Rn = self._nstep_R
            if not d_last and len(self._nstep_queue) >= self.n_step:
                # nichts weiter zu tun: Rn hat bereits n Schritte (inkl. Discount)
                pass
            # Schreibe fertigen Übergang
            self.buffer.append((s0, a0, Rn, d_last, s_last, pos0, mask0, mask_last))

            # Slide-Fenster: entferne die erste Transition und aktualisiere Akkus
            s_pop, a_pop, r_pop, d_pop, _, _, _, _ = self._nstep_queue.popleft()
            self._nstep_R = (self._nstep_R - float(r_pop)) / self.gamma  # inverse der vorherigen Update-Formel
            # Achtung: korrekte Rückrechnung des Discounts:
            # vorher: R += gamma_pow * r; gamma_pow *= gamma
            # für Pop: (R - r0) / gamma
            self._nstep_gamma /= self.gamma

        # Bei Episodenende: restliche Schritte in Fenster ohne weiteres Bootstrap schreiben
        if done:
            self._flush_nstep_tail()

    def _flush_nstep_tail(self):
        """Flush aller verbleibenden N-step Übergänge ohne weiteres Bootstrap (Ende Episode)."""
        while len(self._nstep_queue) > 0:
            # Rechne aktuellen akkumulierten Return von der Queue neu (ohne Bootstrap)
            R = 0.0
            g = 1.0
            for _, _, r, _, _, _, _, _ in self._nstep_queue:
                R += g * float(r)
                g *= self.gamma

            s0, a0, _, d0, _, pos0, mask0, _ = self._nstep_queue[0]
            _, _, _, d_last, s_last, _, _, mask_last = self._nstep_queue[-1]
            self.buffer.append((s0, a0, R, d_last, s_last, pos0, mask0, mask_last))
            self._nstep_queue.popleft()

        self._nstep_R = 0.0
        self._nstep_gamma = 1.0

    def _flush_episode_lambda(self):
        """
        Berechnet G^λ für die gesammelte Episode und schreibt 1-Step-ähnliche Tupel in self.buffer:
        (s_t, a_t, G^λ_t, done_t', s_{t+1..}, pos_t, mask_t, mask_{t+1..})
        Bootstrap über value_fn am jeweiligen Folgezustand.
        """
        assert self.value_fn is not None, "Für λ-Returns muss value_fn gesetzt sein."

        ep = self._episode  # Kurzname
        T = len(ep)

        # V_{t+1} für alle t bestimmen (Bootstrap-Ziel am Folgezustand)
        # Terminalzustände erhalten V=0
        V_next = np.zeros((T,), dtype=np.float32)
        for t, (_, _, _, done, next_state, unit_pos, _, next_mask) in enumerate(ep):
            if not done:
                V_next[t] = float(self.value_fn(next_state, next_mask, unit_pos))
            else:
                V_next[t] = 0.0

        # Rückwärts-Rekursion für G^λ:
        #   G_T^λ = r_T + γ * V_{T+1}    (falls letzter Schritt nicht terminal — sonst ohne Bootstrap)
        #   G_t^λ = r_t + γ * ((1-λ) * V_{t+1} + λ * G_{t+1}^λ)
        G_lam = np.zeros((T,), dtype=np.float32)
        for t in reversed(range(T)):
            s, a, r, done, next_state, pos, mask, next_mask = ep[t]
            if done:
                G_lam[t] = float(r)  # am Terminus kein Bootstrap
            else:
                bootstrap = (1.0 - self.lam) * V_next[t] + self.lam * (G_lam[t+1] if t+1 < T else V_next[t])
                G_lam[t] = float(r) + self.gamma * bootstrap

        # Schreibe die Episoden-Übergänge als (s_t, a_t, G^λ_t, done_t, next_state_t, ...)
        for t in range(T):
            s, a, r, done, next_state, pos, mask, next_mask = ep[t]
            self.buffer.append((s, a, float(G_lam[t]), done, next_state, pos, mask, next_mask))

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
        raw_masks_np = self.env.venv.venv.get_action_mask()
        raw_masks = torch.from_numpy(raw_masks_np).to(device=device).bool() # [num_envs, H*W, 78]
        _, h, w, _ = self.state.shape
        num_envs = self.env.num_envs


        full_action = np.zeros((num_envs, h, w, 7), dtype=np.int32)
        state_v = torch.tensor(self.state.transpose(0, 3, 1, 2), dtype=torch.float32, device=device)

        def sample_valid(mask):
            """ Gibt einen zufällig ausgewählten Index zurück, bei dem die Eingabemaske True ist."""
            idx = torch.where(mask)[0]
            return idx[torch.randint(len(idx), (1,))] if len(idx) > 0 else torch.tensor(0, device=device)

        for env_i in range(num_envs):
            for i in range(h):
                for j in range(w):
                    if self.state[env_i, i, j, 11] == 1 and self.state[env_i, i, j, 21] == 1:
                        flat_idx = i * w + j                                                                            # wird hierndie richtige Zelle ausgewählt, stimmt der Flat:index?
                        cell_mask = self.convert_78_to_89_mask(raw_masks[env_i, flat_idx])
                        # --- Aktionsauswahl ---
                        if np.random.random() < epsilon:
                            #Originale Maske für die gültigen Action Types
                            a_type = sample_valid(raw_masks[env_i, flat_idx][0:6]).item()
                            full_action[env_i, i, j, 0] = a_type
                            #Mit konvertierter Maske
                            if a_type == 1:
                                single_action= sample_valid(cell_mask[0:4]).item()
                            elif a_type == 2:
                                single_action = sample_valid(cell_mask[4:8]).item()
                            elif a_type == 3:
                                single_action = sample_valid(cell_mask[8:12]).item()
                            elif a_type == 4:
                                single_action = sample_valid(cell_mask[12:40]).item()
                            elif a_type == 5:
                                single_action = sample_valid(cell_mask[40:89]).item()
                            else:
                                single_action = sample_valid(cell_mask).item()
                            full_action[env_i, i, j]=self.qval_to_action(single_action)


                        else:

                            # Eingabe vorbereiten für genau eine Unit:
                            state_v_single = state_v[env_i:env_i + 1]  # Form: [1, C, H, W]
                            unit_pos = torch.tensor([[j, i]], dtype=torch.float32, device=device)  # Form: [1, 2]

                            # Netz aufrufen
                            q_vals_v = net(state_v_single, unit_pos=unit_pos)[0]  # jetzt Shape: [89]

                            # Maskierung anwenden
                            masked_q_vals = q_vals_v.masked_fill(~cell_mask, -1e9)

                            # Beste gültige Aktion auswählen
                            q_val = torch.argmax(masked_q_vals).item()

                            # Umwandlung in Aktionsarray
                            full_action[env_i, i, j] = self.qval_to_action(q_val)


        # --- Schritt ausführen ---
        prev_state = self.state.copy()
        new_state, reward, is_done, infos = self.env.step(full_action.reshape(self.env.num_envs, -1))
        #envs.venv.venv.render(mode="human")
        next_raw_masks = self.env.venv.venv.get_action_mask()

        # --- Replay Buffer befüllen ---
        for env_i in range(num_envs):
            for i in range(h):
                for j in range(w):
                    if self.state[env_i, i, j, 11] == 1 and self.state[env_i, i, j, 21] == 1:
                        flat_idx = i * h + j
                        action_mask = self.convert_78_to_89_mask(raw_masks[env_i, flat_idx])
                        next_action_mask = self.convert_78_to_89_mask(next_raw_masks[env_i, flat_idx])
                        single_action = np.array(full_action[env_i, i, j], dtype=np.int64)
                        single_action = self.action_to_qval(single_action)

                        self.exp_buffer.append(
                            prev_state[env_i],
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

                steps = self.episode_steps[env_i]
                raw_stats = infos[env_i].get("microrts_stats", {})
                episode_info = {
                    "env": env_i,
                    "episode": ep,
                    "reward": shaped,
                    "steps": steps,
                    "epsilon": epsilon
                }

                # Nur rohe Rewards ohne 'discounted'
                for k in ["WinLoss", "ResourceGather", "ProduceWorker", "ProduceBuilding", "Attack",
                          "ProduceCombatUnit"]:
                    episode_info[k] = raw_stats.get(f"{k}RewardFunction", 0.0)

                episode_results.append(episode_info)
                self.env_episode_counter[env_i] += 1
                self.total_rewards[env_i] = 0.0
                self.episode_steps[env_i] = 0

        return {"done": False, "episode_stats": episode_results}

    def _stack_bool_mask(mask_list, device):
        if mask_list is None:
            return None
        out = []
        for m in mask_list:
            if isinstance(m, torch.Tensor):
                out.append(m.to(device).bool())
            else:
                out.append(torch.as_tensor(m, device=device, dtype=torch.bool))
        return torch.stack(out, dim=0)  # [B, A]


    def calc_loss(self, batch, tgt_net, gamma: float):
        states, actions, rewards, dones, next_states, unit_positions, action_masks, next_action_masks = batch
        device = self.device
        B = len(actions)

        # ---- Tensor-Prep ----
        states_v = torch.as_tensor(states, dtype=torch.float32, device=device).permute(0, 3, 1, 2)  # [B,C,H,W]
        next_states_v = torch.as_tensor(next_states, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
        actions_v = torch.as_tensor(actions, dtype=torch.int64, device=device)  # [B]
        rewards_v = torch.as_tensor(rewards, dtype=torch.float32, device=device)  # [B]
        done_mask = torch.as_tensor(dones, dtype=torch.bool, device=device)  # [B]
        unit_pos = torch.as_tensor(unit_positions, dtype=torch.long, device=device)  # [B,2]

        # ---- Q(s,·) und Q(s′,·) ----
        qvals = self.net(states_v, unit_pos=unit_pos)  # [B, 89]
        with torch.no_grad():
            qvals_next = tgt_net(next_states_v, unit_pos=unit_pos)  # [B, 89]  (falls du next_pos hast: hier einsetzen)

        # ---- Q(s,a) extrahieren ----
        state_action_qvals = qvals[torch.arange(B, device=device), actions_v]  # [B]

        # ---- max_a Q(s′,a) mit Maskierung ----
        next_mask = self._stack_bool_mask(next_action_masks, device)  # [B,89] oder None
        if next_mask is not None:
            masked_next = qvals_next.masked_fill(~next_mask, -1e9)
            max_next_q, _ = masked_next.max(dim=1)  # [B]
            # Edge case: falls in einer Zeile gar kein True: setze 0
            any_valid = next_mask.any(dim=1)
            max_next_q = torch.where(any_valid, max_next_q, torch.zeros_like(max_next_q))
        else:
            max_next_q, _ = qvals_next.max(dim=1)

        # ---- Targets je nach Return-Typ ----
        if getattr(self, "use_lambda", False):
            # rewards_v == G^λ_t (bereits mit Bootstrap gemischt) -> kein weiteres Bootstrap!
            target_qvals = rewards_v
        else:
            n = int(getattr(self, "n_step", 1))
            gamma_pow = (gamma ** n) if n > 1 else gamma
            bootstrap = (~done_mask).float() * gamma_pow * max_next_q
            target_qvals = rewards_v + bootstrap

        # ---- Huber-Loss ----
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
    reward_weights = np.array([50.0, 3.0, 3.0, 0.0, 5.0, 1.0])
    print("Reward Weights:", reward_weights)
    num_envs = args.num_bot_envs
    num_each = num_envs // 4  # ganzzahliger Anteil
    envs = MicroRTSGridModeVecEnv(
        num_selfplay_envs=args.num_selfplay_envs,
        num_bot_envs=args.num_bot_envs,
        partial_obs=args.partial_obs,
        max_steps=args.max_steps,
        render_theme=2,
        ai2s= ( [microrts_ai.passiveAI for _ in range(num_each)] +
                [microrts_ai.workerRushAI for _ in range(num_each)] +
                [microrts_ai.lightRushAI for _ in range(num_each)] +
                [microrts_ai.coacAI for _ in range(num_envs - 3 * num_each)]) ,
        map_paths=[args.train_maps[0]],
        reward_weight=reward_weights,
        # Win, Ressource, ProduceWorker, Produce Building, Attack, ProduceCombat Unit, (auskommentiert closer to enemy base)
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

    """
    Initialisierung
    """
    dummy_obs = envs.reset()
    state_shape = dummy_obs.shape[1:]  # [H, W, C]
    action_shape = (7,)  # [H, W, 7] später
    expbuffer = ReplayBuffer(capacity=args.buffer_memory, state_shape=state_shape, action_shape=action_shape, n_step=5)
    dummy_input_shape = (29, 8, 8)  # [C, H, W]
    policy_net = UASDRQN(input_shape=dummy_input_shape).to(device)
    target_net = UASDRQN(input_shape=dummy_input_shape).to(device)
    target_net.load_state_dict(policy_net.state_dict())  #  Initiales Sync
    target_net.eval()
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.learning_rate)
    agent = Agent(env=envs, exp_buffer=expbuffer, net=policy_net, device=device)

    frame_idx = 0
    episode_idx = 0
    best_mean_reward = None
    reward_queue = deque(maxlen=100)
    warmup_frames = args.warmup_frames
    eval_interval = 100000
    frame_start = 0

    # Parameterzähler
    total_params = sum(p.numel() for p in policy_net.parameters() if p.requires_grad)
    print(f"Gesamtanzahl der trainierbaren Parameter: {total_params}")

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
    torch.save(policy_net.state_dict(), f"./{args.exp_name}/{args.exp_name}_initial.pth")
    # Training
    while frame_idx < args.total_timesteps:
        frame_idx += 1
        if frame_idx % 10000 == 0:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Frame idx: {frame_idx}")
        epsilon = max(args.epsilon_final, args.epsilon_start - frame_idx / args.epsilon_decay)
        if frame_idx < warmup_frames:
            epsilon = 1.0
        log_path = f"./{args.exp_name}/{args.exp_name}_train_log.csv"
        file_exists = os.path.exists(log_path)
        step_info = agent.play_step(epsilon=epsilon)
        for ep_data in step_info.get("episode_stats", []):
            with open(log_path, "a") as f:
                # Frame-Index hinzufügen
                ep_data_with_frame = dict(ep_data)
                ep_data_with_frame["frame_idx"] = frame_idx  #ergänzt zu den epsisoden infos noch den aktuellen Frame_idx

                if not file_exists or os.stat(log_path).st_size == 0:
                    header = list(ep_data_with_frame.keys())
                    f.write(",".join(header) + "\n")
                else:
                    header = list(ep_data_with_frame.keys())  # trotzdem notwendig

                values = [str(ep_data_with_frame[k]) for k in header]
                f.write(",".join(values) + "\n")

        if len(expbuffer) < args.batch_size:
            continue
        # Target-Sync
        if frame_idx % args.sync_interval == 0:
            target_net.load_state_dict(policy_net.state_dict())
        # Checkpoint speichern
        if frame_idx % args.save_network == 0:
            save_path = os.path.join(model_dir, f"{args.exp_name}_{frame_idx}.pth")
            torch.save(policy_net.state_dict(), save_path)
            print(f"Checkpoint gespeichert: {save_path}")
        # Training
        batch = expbuffer.sample(args.batch_size)
        optimizer.zero_grad()
        loss = agent.calc_loss(batch, target_net, gamma=args.gamma)
        loss.backward()
        optimizer.step()
    # Training fertig – final speichern
    torch.save(policy_net.state_dict(), f"./{args.exp_name}/{args.exp_name}_final.pth")
    print("Training abgeschlossen.")

    envs.close()





