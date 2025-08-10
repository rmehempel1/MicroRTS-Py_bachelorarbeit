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

import torch
import torch.nn as nn
import torch.nn.functional as F

class UASDRQN(nn.Module):
    def __init__(self, input_shape, hidden_size=256, gru_layers=1, pos_dim=32, q_dim=89):
        super().__init__()
        self.in_channels, self.h ,self.w = input_shape
        self.hidden_size  = hidden_size
        self.gru_layers   = gru_layers
        self.q_dim        = q_dim

        # --- Encoder (beliebig; Beispiel) ---
        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # berechne Feature-Länge F_out
        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, self.h, self.w)
            f = self.encoder(dummy).numel()
        self.feat_lin = nn.Linear(f, hidden_size)  # in GRU-Eingabe projizieren

        # --- reine Zustands-Rekurrenz (ohne Position!) ---
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=gru_layers, batch_first=True)

        # --- Positions-Embedding + Head ---
        self.pos_mlp = nn.Sequential(
            nn.Linear(2, pos_dim), nn.ReLU(),
            nn.Linear(pos_dim, pos_dim), nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size + pos_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, q_dim)
        )

    # Hilfslayer: normiere Gitterkoordinaten
    def _norm_pos(self, pos_xy, device):
        # pos_xy: [..., 2] in (x=j, y=i)
        x = pos_xy[..., 0] / max(1, (self.w  - 1))
        y = pos_xy[..., 1] / max(1, (self.h - 1))
        return torch.stack([x, y], dim=-1).to(device).float()

    @torch.no_grad()
    def initial_state(self, B, device):
        return torch.zeros(self.gru_layers, B, self.hidden_size, device=device)

    def context(self, x5d, h0=None):
        """
        x5d: [B, T, C, H, W]
        return: ctx [B, T, H], h_n [L, B, H]
        """
        assert x5d.dim() == 5, "erwarte [B,T,C,H,W]"
        B, T, C, H, W = x5d.shape
        x = x5d.reshape(B*T, C, H, W)
        feat = self.encoder(x).reshape(B*T, -1)
        feat = self.feat_lin(feat).relu()
        feat = feat.view(B, T, -1)              # [B,T,H]
        if h0 is None:
            h0 = self.initial_state(B, x5d.device)
        out_seq, h_n = self.gru(feat, h0)       # [B,T,H], [L,B,H]
        return out_seq, h_n

    def q_from_context(self, ctx_bt, unit_pos_bt):
        """
        ctx_bt: [B, T, H]   (oder [N, 1, H] wenn wir nur T=1 haben und N Units batchen)
        unit_pos_bt: [B, T, 2] (oder [N,1,2])
        """
        B, T, H = ctx_bt.shape
        ctx = ctx_bt.reshape(B*T, H)            # [B*T,H]
        pos = self._norm_pos(unit_pos_bt.reshape(B*T, 2), ctx_bt.device)  # [B*T,2]
        pos_e = self.pos_mlp(pos)               # [B*T,P]
        x = torch.cat([ctx, pos_e], dim=-1)     # [B*T,H+P]
        q = self.head(x).view(B, T, self.q_dim) # [B,T,89]
        return q

    def forward(self, x, unit_pos=None, h0=None):
        """
        Convenience: akzeptiert 4D ([B,C,H,W]) oder 5D.
        Wenn 4D, wird T=1 angenommen. unit_pos optional; wenn None -> (0,0).
        """
        if x.dim() == 4:
            x = x.unsqueeze(1)  # [B,1,C,H,W]
            if unit_pos is not None and unit_pos.dim() == 2:
                unit_pos = unit_pos.unsqueeze(1)  # [B,1,2]
        assert x.dim() == 5
        B, T, *_ = x.shape
        if unit_pos is None:
            unit_pos = x.new_zeros(B, T, 2)

        ctx, h_n = self.context(x, h0)
        q = self.q_from_context(ctx, unit_pos)  # [B,T,89]
        return q, h_n



import numpy as np
from collections import deque

from collections import deque
from typing import Any, Callable, Deque, Optional, Tuple, List
import numpy as np


class ReplayBuffer:
    """
    Replay-Buffer mit Unterstützung für:
      - 1-Step (Standard)
      - N-Step-Returns (online)
      - λ-Returns (episodenweise; benötigt value_fn)

    RNN-fähig:
      - Optionales Mitspeichern von Hidden-States (hidden_t, hidden_tp1)
      - Optionales Mitspeichern der Folg(e)position der handelnden Unit (unit_pos_next)

    Erwartete value_fn-Signatur (flexibel, optionale Argumente):
        value_fn(next_state, next_action_mask, unit_pos_next=None, hidden_tp1=None) -> float

    Gespeichertes Transition-Format (Tuple):
        (state, action, return_or_reward, done, next_state,
         unit_pos, action_masks, next_action_masks,
         hidden_t, hidden_tp1, unit_pos_next)

    Hinweise:
      - Bei N-Step wird 'return_or_reward' zum N-Step-Return R_n.
      - Bei λ-Returns wird 'return_or_reward' zu G^λ_t.
      - Für 1-Step ist es schlicht r_t (oder R_1).

    Parameter:
      capacity     : max. Größe des Hauptpuffers (Ringpuffer)
      state_shape  : Form von state (z.B. [H, W, C]) – informativ, keine Striktprüfung
      action_shape : Form von action (z.B. [7] bei MicroRTS) – informativ
      gamma        : Diskontfaktor
      n_step       : N für N-Step-Returns (n_step=1 => 1-Step)
      use_lambda   : True => nutze episodenweise λ-Returns
      lam          : λ in [0,1]
      value_fn     : Callable zur Bootstrap-Schätzung V(s_{t+1} | optional hidden_tp1)
      store_hidden : Ob hidden_t/hidden_tp1 abgelegt werden sollen (True empfohlen bei RNN)
    """

    def __init__(self,
                 capacity: int,
                 state_shape: Tuple[int, ...],
                 action_shape: Tuple[int, ...],
                 gamma: float = 0.99,
                 n_step: int = 1,
                 use_lambda: bool = False,
                 lam: float = 0.95,
                 value_fn: Optional[Callable[..., float]] = None,
                 store_hidden: bool = True):
        self.buffer: Deque[tuple] = deque(maxlen=int(capacity))
        self.state_shape = tuple(state_shape)
        self.action_shape = tuple(action_shape)

        # Rückgabe-Optionen
        self.gamma = float(gamma)
        self.n_step = int(max(1, n_step))
        self.use_lambda = bool(use_lambda)
        self.lam = float(lam)
        self.value_fn = value_fn
        self.store_hidden = bool(store_hidden)

        # Für N-step Aggregation (online)
        # enthält (s,a,r,done,s_next,pos,mask,mask_next,h_t,h_tp1,pos_next)
        self._nstep_queue: Deque[tuple] = deque()
        self._nstep_R: float = 0.0
        self._nstep_gamma: float = 1.0

        # Für λ-Returns (Episodenpuffer)
        # list of transitions wie oben
        self._episode: List[tuple] = []

    def __len__(self) -> int:
        return len(self.buffer)

    # ----------- öffentliche API -----------

    def append(self,
               state: np.ndarray,
               action: np.ndarray,
               reward: float,
               done: bool,
               next_state: np.ndarray,
               unit_pos: Tuple[int, int],
               action_masks: Any,
               next_action_masks: Any,
               hidden_t: Optional[Any] = None,
               hidden_tp1: Optional[Any] = None,
               unit_pos_next: Optional[Tuple[int, int]] = None) -> None:
        """
        Fügt eine Transition hinzu. Je nach Modus (λ vs. n-step) wird sie
        direkt in den Hauptpuffer aggregiert oder zunächst gesammelt.
        """
        # Falls Hidden-States nicht gespeichert werden sollen, auf None setzen
        if not self.store_hidden:
            hidden_t = None
            hidden_tp1 = None

        tr = (state, action, float(reward), bool(done), next_state,
              unit_pos, action_masks, next_action_masks,
              hidden_t, hidden_tp1, unit_pos_next)

        if self.use_lambda:
            self._episode.append(tr)
            if done:
                self._flush_episode_lambda()
                self._episode.clear()
        else:
            self._push_nstep(tr)

    def sample(self, batch_size: int):
        """
        Zieht zufällige Batches ohne Zurücklegen.
        Gibt Arrays für states/actions/... zurück; Masken/Hidden-States werden als
        Liste/Struktur zurückgegeben, um heterogene Shapes zu erlauben.
        """
        assert len(self.buffer) >= batch_size, "ReplayBuffer: zu wenige Samples."

        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        samples = [self.buffer[i] for i in idx]

        (states, actions, returns, dones, next_states,
         unit_positions, action_masks, next_action_masks,
         hidden_t, hidden_tp1, unit_pos_next) = zip(*samples)

        states = np.array(states, copy=False)                 # [B, ...state_shape]
        actions = np.stack(actions)                           # [B, ...action_shape]
        returns = np.asarray(returns, dtype=np.float32)       # [B]
        dones = np.asarray(dones, dtype=np.bool_)             # [B]
        next_states = np.array(next_states, copy=False)       # [B, ...state_shape]
        unit_positions = np.asarray(unit_positions, dtype=np.int32)  # [B, 2]
        # action_masks / next_action_masks heterogen → als Tuple zurückgeben
        unit_pos_next = np.asarray(unit_pos_next, dtype=object) if any(p is not None for p in unit_pos_next) else None

        # Hidden-States ggf. als Liste/Tuple zurückgeben (keine Annahmen über Typ/Shape)
        hidden_t = list(hidden_t)
        hidden_tp1 = list(hidden_tp1)

        return (states, actions, returns, dones, next_states,
                unit_positions, action_masks, next_action_masks,
                hidden_t, hidden_tp1, unit_pos_next)

    def finalize_pending(self) -> None:
        """
        Am Episoden-/Trainingsende aufrufen, um offene Aggregationen zu flushen.
        """
        if self.use_lambda and len(self._episode) > 0:
            self._flush_episode_lambda()
            self._episode.clear()
        self._flush_nstep_tail()

    # ----------- interne Helfer -----------

    def _push_nstep(self, tr: tuple) -> None:
        """
        Online N-step Aggregation. Schreibt fertige N-step-Übergänge in self.buffer.
        """
        (s, a, r, done, s_next, pos, mask, mask_next, h_t, h_tp1, pos_next) = tr
        self._nstep_queue.append(tr)

        # Akkumuliere Return und Discount (laufendes Fenster)
        self._nstep_R += self._nstep_gamma * float(r)
        self._nstep_gamma *= self.gamma

        # Wenn wir n Schritte beisammen haben oder die Episode endet → schreiben
        if len(self._nstep_queue) >= self.n_step or done:
            # Kopf-Transition (Zeit t)
            (s0, a0, _, d0, _, pos0, mask0, _, h0, _, _) = self._nstep_queue[0]
            # Letzter Eintrag (Zeit t+k-1)
            (_, _, _, d_last, s_last, _, _, mask_last, _, h_last, pos_last_next) = self._nstep_queue[-1]

            Rn = self._nstep_R  # bereits korrekt diskontiert für die im Fenster liegenden Rewards
            # Kein weiteres Bootstrap hier: Bootstrapping geschieht später bei Target-Berechnung über value_fn/Q_target.

            self.buffer.append((
                s0, a0, Rn, d_last, s_last,
                pos0, mask0, mask_last,
                h0, h_last, pos_last_next
            ))

            # Fenster nach vorn schieben (erste Transition entfernen) + Akkus anpassen
            _, _, r_pop, _, _, _, _, _, _, _, _ = self._nstep_queue.popleft()
            # inverse der Update-Formel: vorher R += g*r; g*=gamma
            self._nstep_R = (self._nstep_R - float(r_pop)) / self.gamma
            self._nstep_gamma /= self.gamma

        # Bei Episodenende: Rest ohne weiteres Bootstrap flushen
        if done:
            self._flush_nstep_tail()

    def _flush_nstep_tail(self) -> None:
        """
        Flush aller verbleibenden N-step Übergänge ohne weiteres Bootstrap (Ende Episode).
        """
        while len(self._nstep_queue) > 0:
            # Rechne aktuellen akkumulierten Return der Queue neu (ohne Bootstrap)
            R = 0.0
            g = 1.0
            for _, _, r, _, _, _, _, _, _, _, _ in self._nstep_queue:
                R += g * float(r)
                g *= self.gamma

            (s0, a0, _, d0, _, pos0, mask0, _, h0, _, _) = self._nstep_queue[0]
            (_, _, _, d_last, s_last, _, _, mask_last, _, h_last, pos_last_next) = self._nstep_queue[-1]

            self.buffer.append((
                s0, a0, R, d_last, s_last,
                pos0, mask0, mask_last,
                h0, h_last, pos_last_next
            ))
            self._nstep_queue.pop() if len(self._nstep_queue) == 1 else self._nstep_queue.popleft()

        self._nstep_R = 0.0
        self._nstep_gamma = 1.0

    def _flush_episode_lambda(self) -> None:
        """
        Berechnet G^λ für die gesammelte Episode und schreibt 1-Step-ähnliche Tupel in den Hauptpuffer:
            (s_t, a_t, G^λ_t, done_t, s_{t+1}, pos_t, mask_t, mask_{t+1}, h_t, h_{t+1}, unit_pos_next_t)

        Bootstrap über value_fn am jeweiligen Folgezustand. Terminals erhalten V=0.
        """
        assert self.value_fn is not None, "Für λ-Returns muss value_fn gesetzt sein."

        ep = self._episode
        T = len(ep)
        if T == 0:
            return

        # V_{t+1} für alle t bestimmen (Bootstrap-Ziel am Folgezustand)
        V_next = np.zeros((T,), dtype=np.float32)
        for t, (_, _, _, done, next_state, _, _, next_mask, _, hidden_tp1, unit_pos_next) in enumerate(ep):
            if not done:
                # value_fn darf optionale Argumente ignorieren; wir übergeben sie best-effort
                try:
                    V_next[t] = float(self.value_fn(next_state, next_mask,
                                                    unit_pos_next=unit_pos_next,
                                                    hidden_tp1=hidden_tp1))
                except TypeError:
                    # Fallback auf alte Signatur
                    V_next[t] = float(self.value_fn(next_state, next_mask, unit_pos_next))
            else:
                V_next[t] = 0.0

        # Rückwärts-Rekursion für G^λ:
        #   G_T^λ = r_T + γ * V_{T+1} (falls letzter Schritt nicht terminal; sonst nur r_T)
        #   G_t^λ = r_t + γ * ((1-λ) * V_{t+1} + λ * G_{t+1}^λ)
        G_lam = np.zeros((T,), dtype=np.float32)
        for t in reversed(range(T)):
            s, a, r, done, next_state, pos, mask, next_mask, h_t, h_tp1, pos_next = ep[t]
            if done:
                G_lam[t] = float(r)
            else:
                bootstrap = (1.0 - self.lam) * V_next[t] + self.lam * (G_lam[t + 1] if t + 1 < T else V_next[t])
                G_lam[t] = float(r) + self.gamma * bootstrap

        # Schreibe Episoden-Übergänge
        for t in range(T):
            s, a, r, done, s_next, pos, mask, mask_next, h_t, h_tp1, pos_next = ep[t]
            self.buffer.append((s, a, float(G_lam[t]), done, s_next,
                                pos, mask, mask_next, h_t, h_tp1, pos_next))


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

    def _init_hidden(self, batch_size: int, device):
        """
        Erstellt einen Null-Hidden-State passend zu deinem Netz.
        Unterstützt GRU/LSTM. Greift bevorzugt auf net.rnn.* zu.
        Fällt sonst auf (num_layers=1, hidden_size=256) zurück – passe das ggf. an.
        """
        net = self.net
        # Versuche, Parameter vom eingebauten RNN zu lesen
        rnn = getattr(net, "rnn", None)
        num_layers = getattr(net, "num_layers", None)
        hidden_size = getattr(net, "hidden_size", None)

        if rnn is not None:
            num_layers = getattr(rnn, "num_layers", num_layers)
            hidden_size = getattr(rnn, "hidden_size", hidden_size)
            is_lstm = isinstance(rnn, nn.LSTM)
        else:
            # Heuristik/Fallback
            is_lstm = bool(getattr(net, "is_lstm", False))

        if num_layers is None or hidden_size is None:
            # <<< WICHTIG: Falls du die echten Werte kennst, trage sie hier ein!
            num_layers = 1
            hidden_size = 256

        # Bidirectional? (falls ja, *2 auf die erste Dimension)
        num_directions = 2 if (rnn is not None and getattr(rnn, "bidirectional", False)) else 1
        layers = num_layers * num_directions

        h0 = torch.zeros(layers, batch_size, hidden_size, device=device)
        if is_lstm:
            c0 = torch.zeros_like(h0)
            return (h0, c0)
        else:
            return h0

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

    def play_step(self, epsilon: float = 0.0):
        net = self.net
        device = self.device
        episode_results = []

        # --- Masken & Shapes ---
        raw_masks_np = self.env.venv.venv.get_action_mask()  # [num_envs, H*W, 78]
        raw_masks = torch.from_numpy(raw_masks_np).to(device=device).bool()
        _, h, w, _ = self.state.shape
        num_envs = self.env.num_envs

        # --- Hidden-State initialisieren ---
        if not hasattr(self, "rnn_state") or self.rnn_state is None:
            self.rnn_state = self._init_hidden(batch_size=num_envs, device=device)

        # Aktionen-Array
        full_action = np.zeros((num_envs, h, w, 7), dtype=np.int32)
        next_rnn_state_list = [None] * num_envs

        @torch.no_grad()
        def sample_valid(mask: torch.Tensor) -> torch.Tensor:
            idx = torch.where(mask)[0]
            if idx.numel() == 0:
                return torch.tensor(0, device=device)
            ridx = torch.randint(idx.numel(), (1,), device=device)
            return idx[ridx]

        net.eval()
        with torch.no_grad():
            for env_i in range(num_envs):
                # Hidden-State extrahieren
                if isinstance(self.rnn_state, tuple):
                    rnn_in = (
                        self.rnn_state[0][:, env_i:env_i + 1, :].contiguous(),
                        self.rnn_state[1][:, env_i:env_i + 1, :].contiguous()
                    )
                else:
                    rnn_in = self.rnn_state[:, env_i:env_i + 1, :].contiguous()

                # State als [1,1,C,H,W]
                state_v_single_5d = torch.tensor(
                    self.state[env_i].transpose(2, 0, 1),
                    dtype=torch.float32,
                    device=device
                ).unsqueeze(0).unsqueeze(1)

                # 1) einmaliger RNN-Schritt -> Kontext
                ctx, h_next = net.context(state_v_single_5d, h0=rnn_in)
                next_rnn_state_list[env_i] = h_next

                # 2) aktive Units finden
                pos_list, idx_list = [], []
                for i in range(h):
                    for j in range(w):
                        if self.state[env_i, i, j, 11] == 1 and self.state[env_i, i, j, 21] == 1:
                            pos_list.append([j, i])
                            idx_list.append((i, j))
                if len(pos_list) == 0:
                    continue

                pos_bt = torch.tensor(pos_list, dtype=torch.float32, device=device).unsqueeze(1)
                ctx_bt = ctx.repeat(len(pos_list), 1, 1)

                # 3) Q-Werte
                q_all = net.q_from_context(ctx_bt, pos_bt)[:, 0, :]

                # 4) Maskieren + Aktion auswählen
                for k, (i, j) in enumerate(idx_list):
                    flat_idx = i * w + j
                    cell_mask = self.convert_78_to_89_mask(raw_masks[env_i, flat_idx])

                    if np.random.random() < epsilon:
                        a_type = sample_valid(raw_masks[env_i, flat_idx][0:6]).item()
                        full_action[env_i, i, j, 0] = a_type
                        if a_type == 1:
                            single_action = sample_valid(cell_mask[0:4]).item()
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
                    else:
                        q_vals = q_all[k].masked_fill(~cell_mask, -1e9)
                        single_action = torch.argmax(q_vals).item()

                    full_action[env_i, i, j] = self.qval_to_action(single_action)

        # --- Schritt ausführen ---
        prev_state = self.state.copy()
        new_state, reward, is_done, infos = self.env.step(
            full_action.reshape(self.env.num_envs, -1)
        )
        next_raw_masks = self.env.venv.venv.get_action_mask()

        # --- Replay Buffer befüllen ---
        for env_i in range(num_envs):
            if isinstance(self.rnn_state, tuple):
                hidden_t = (self.rnn_state[0][:, env_i:env_i + 1, :].detach().cpu(),
                            self.rnn_state[1][:, env_i:env_i + 1, :].detach().cpu())
                hidden_tp1 = (next_rnn_state_list[env_i][0].detach().cpu(),
                              next_rnn_state_list[env_i][1].detach().cpu())
            else:
                hidden_t = self.rnn_state[:, env_i:env_i + 1, :].detach().cpu()
                hidden_tp1 = next_rnn_state_list[env_i].detach().cpu()

            for i in range(h):
                for j in range(w):
                    if self.state[env_i, i, j, 11] == 1 and self.state[env_i, i, j, 21] == 1:
                        flat_idx = i * w + j
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
                            next_action_mask,
                            hidden_t,
                            hidden_tp1
                        )

        # --- Hidden-State übernehmen ---
        def detach_hidden(h):
            if isinstance(h, tuple):
                return (h[0].detach(), h[1].detach())
            return h.detach()

        if isinstance(self.rnn_state, tuple):
            h_list, c_list = [], []
            for env_i in range(num_envs):
                h_i, c_i = next_rnn_state_list[env_i]
                if is_done[env_i]:
                    h_i = torch.zeros_like(h_i)
                    c_i = torch.zeros_like(c_i)
                h_list.append(h_i)
                c_list.append(c_i)
            self.rnn_state = (torch.cat(h_list, dim=1), torch.cat(c_list, dim=1))
            self.rnn_state = (detach_hidden(self.rnn_state[0]), detach_hidden(self.rnn_state[1]))
        else:
            h_list = []
            for env_i in range(num_envs):
                h_i = next_rnn_state_list[env_i]
                if is_done[env_i]:
                    h_i = torch.zeros_like(h_i)
                h_list.append(h_i)
            self.rnn_state = torch.cat(h_list, dim=1)
            self.rnn_state = detach_hidden(self.rnn_state)

        # --- Episodenabschluss / Logging ---
        self.state = new_state
        for env_i in range(num_envs):
            self.total_rewards[env_i] += reward[env_i]
            self.episode_steps[env_i] += 1

        for env_i in range(num_envs):
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
                for k in ["WinLoss", "ResourceGather", "ProduceWorker", "ProduceBuilding", "Attack",
                          "ProduceCombatUnit"]:
                    episode_info[k] = raw_stats.get(f"{k}RewardFunction", 0.0)

                episode_results.append(episode_info)
                self.env_episode_counter[env_i] += 1
                self.total_rewards[env_i] = 0.0
                self.episode_steps[env_i] = 0

        return {
            "done": any(is_done),
            "episode_stats": episode_results
        }

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
        (states, actions, rewards, dones, next_states,
         unit_positions, action_masks, next_action_masks,
         hidden_t, hidden_tp1, unit_pos_next) = batch
        device = self.device
        B = len(actions)

        # ---- Tensor-Prep ----
        states_v = torch.as_tensor(states, dtype=torch.float32, device=device).permute(0, 3, 1, 2)  # [B,C,H,W]
        next_states_v = torch.as_tensor(next_states, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
        actions_v = torch.as_tensor(actions, dtype=torch.int64, device=device)  # [B]
        rewards_v = torch.as_tensor(rewards, dtype=torch.float32, device=device)  # [B]
        done_mask = torch.as_tensor(dones, dtype=torch.bool, device=device)  # [B]
        unit_pos = torch.as_tensor(unit_positions, dtype=torch.long, device=device)  # [B,2]

        # Falls unit_pos_next existiert und nicht None → umwandeln
        if unit_pos_next is not None:
            unit_pos_next_v = torch.as_tensor(unit_pos_next, dtype=torch.long, device=device)
        else:
            unit_pos_next_v = unit_pos  # fallback: gleiche Position wie vorher

        # ---- Q(s,·) und Q(s′,·) ----
        # ---- Q(s,·) und Q(s′,·) ----
        qvals_out = self.net(states_v, unit_pos=unit_pos)
        if isinstance(qvals_out, tuple):
            qvals = qvals_out[0]  # nur Q-Werte
        else:
            qvals = qvals_out

        with torch.no_grad():
            qvals_next_out = tgt_net(next_states_v, unit_pos=unit_pos_next_v)
            if isinstance(qvals_next_out, tuple):
                qvals_next = qvals_next_out[0]
            else:
                qvals_next = qvals_next_out


        # ---- Q(s,a) extrahieren ----
        state_action_qvals = qvals[torch.arange(B, device=device), actions_v]  # [B]

        # ---- max_a Q(s′,a) mit Maskierung ----
        next_mask = self._stack_bool_mask(next_action_masks, device)  # [B,89] oder None
        if next_mask is not None:
            masked_next = qvals_next.masked_fill(~next_mask, -1e9)
            max_next_q, _ = masked_next.max(dim=1)
            any_valid = next_mask.any(dim=1)
            max_next_q = torch.where(any_valid, max_next_q, torch.zeros_like(max_next_q))
        else:
            max_next_q, _ = qvals_next.max(dim=1)

        # ---- Targets ----
        if getattr(self, "use_lambda", False):
            target_qvals = rewards_v
        else:
            n = int(getattr(self, "n_step", 1))
            gamma_pow = (gamma ** n) if n > 1 else gamma
            bootstrap = (~done_mask).float() * gamma_pow * max_next_q
            target_qvals = rewards_v + bootstrap

        # ---- Loss ----
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





