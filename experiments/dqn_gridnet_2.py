
import argparse
import os
import random
import subprocess
import time
from distutils.util import strtobool
from typing import List
from collections import deque


import numpy as np
import pandas as pd
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
        help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--capture-video', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
        help='whether to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--wandb-project-name', type=str, default="gym-microrts",
        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
        help="the entity (team) of wandb's project")

    # Algorithm specific arguments
    parser.add_argument('--partial-obs', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, the game will have partial observability')
    parser.add_argument('--n-minibatch', type=int, default=4,
        help='the number of mini batch')
    parser.add_argument('--num-bot-envs', type=int, default=0,
        help='the number of bot game environment; 16 bot envs means 16 games')
    parser.add_argument('--num-selfplay-envs', type=int, default=24,
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
        self.gamma = gamma        # gamma ist unser discount faktor

    def reset(self):
        obs = self.venv.reset()
        self.raw_rewards = [[] for _ in range(self.num_envs)]
        self.ts = np.zeros(self.num_envs, dtype=np.float32)
        self.raw_discount_rewards = [[] for _ in range(self.num_envs)]
        return obs

    def step_wait(self):
        """obs reward, dones werden unverändert zurückgegeben und nur infos in newinfos ungewandeld
        """
        obs, rews, dones, infos = self.venv.step_wait()  #observation
        newinfos = list(infos[:])
        for i in range(len(dones)):
            self.raw_rewards[i] += [infos[i]["raw_rewards"]]
            self.raw_discount_rewards[i] += [
                (self.gamma ** self.ts[i])
                * np.concatenate((infos[i]["raw_rewards"], infos[i]["raw_rewards"].sum()), axis=None)
            ]
            self.ts[i] += 1
            if dones[i]:  #wenn Episode zu Ende
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

def test_replay_buffer_once(buffer, expected_shape=(10, 10, 7), batch_size=4):
    """
    Testet den Replay Buffer mit einem Mini-Batch.
    """
    if len(buffer) < batch_size:
        print(f"⚠️ Nicht genug Daten im Buffer ({len(buffer)} von {batch_size} benötigt).")
        return

    states, actions, rewards, dones, next_states = buffer.sample(batch_size)

    print("✅ Replay Buffer Sample erfolgreich:")
    print(f"  states:       {states.shape}")
    print(f"  actions:      {actions.shape}")
    print(f"  rewards:      {rewards.shape}")
    print(f"  dones:        {dones.shape}")
    print(f"  next_states:  {next_states.shape}")

    # Zusätzliche Checks
    assert actions.shape[1:] == expected_shape, f"❌ actions haben unerwartete Form: {actions.shape[1:]}"
    assert states.shape == next_states.shape, "❌ states und next_states inkonsistent"
    assert rewards.shape[0] == batch_size, "❌ rewards falsch geformt"
    print("✅ Formate und Dimensionen stimmen.")


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
        """
        Wählt zufällig ein Mini-Batch von Transitionen aus.
        Gibt Arrays für: states, actions, rewards, dones, next_states zurück.
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, dones, next_states = zip(*batch)

        states = np.array(states)
        next_states = np.array(next_states)
        actions = np.array(actions)

        # Rekonstruiere [B, H, W, 7] aus [B, H*W*7]
        B = actions.shape[0]
        grid_size = int(np.sqrt(actions.shape[1] // 7))  # H = W angenommen
        actions = actions.reshape(B, grid_size, grid_size, 7)

        return (
            states,
            actions,
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.uint8),
            next_states
        )


class MovementHead(nn.Module):
    """Ebenfalls verwendet für Harvest und Return"""

    """Aktuell Getrennte Encoder, um Parameter zu sparen könnten alle den selben Encoder verwenden und anschließen über den Head differenzieren"""
    def __init__(self, in_channels):
        super().__init__()

        self.encoder_decision = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),  # H × W sollte gleich bleiben
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # H × W bleibt gleich
            nn.ReLU()
        )

        self.decision_head = nn.Conv2d(64, 2, kernel_size=1)  # Output Shape für Decision [B, C, H, W], für jedes Grid 0 oder 1 -> C=2 [0,1]

        self.encoder_dir = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),  # H × W sollte gleich bleiben
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # H × W bleibt gleich
            nn.ReLU()
        )

        self.dir_head = nn.Conv2d(64, 4, kernel_size=1)  #C=[0-3]

    def forward(self, x):

        x_decision = self.encoder_decision(x)  # Shape bleibt (B, 64, H, W)
        decision_logits = self.decision_head(x_decision)  # → (B, 2, H, W)

        x_dir = self.encoder_dir(x)  # Shape bleibt (B, 64, H, W)
        dir_logits = self.dir_head(x_dir)  # → (B, 2, H, W)

        return decision_logits, dir_logits


class ProduceHead(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.encoder_decision = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),  # H × W sollte gleich bleiben
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # H × W bleibt gleich
            nn.ReLU()
        )
        self.decision_head = nn.Conv2d(64, 2, kernel_size=1)  # Output Shape für Decision [B, C, H, W], für jedes Grid 0 oder 1 -> C=2 [0,1]

        self.encoder_dir = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),  # H × W sollte gleich bleiben
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # H × W bleibt gleich
            nn.ReLU()
        )
        self.dir_head = nn.Conv2d(64, 4, kernel_size=1)  # C=[0-3]

        self.encoder_type = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),  # H × W sollte gleich bleiben
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # H × W bleibt gleich
            nn.ReLU()
        )
        self.type_head = nn.Conv2d(64, 7, kernel_size=1)  # C=[0-6] resource, base, barrack,worker, light, heavy, ranged

    def forward(self, x):

        x_decision = self.encoder_decision(x)  # Shape bleibt (B, 64, H, W)
        decision_logits = self.decision_head(x_decision)  # → (B, 2, H, W)

        x_dir = self.encoder_dir(x)  # Shape bleibt (B, 64, H, W)
        dir_logits = self.decision_head(x_dir)  # → (B, 2, H, W)

        x_type = self.encoder_type(x)  # Shape bleibt (B, 64, H, W)
        type_logits = self.decision_head(x_type)  # → (B, 2, H, W)

        return decision_logits, dir_logits, type_logits


class AttackHead(nn.Module):
    """Aktuell identisch zum movement head, unklar wie attack direction codiert ist"""
    def __init__(self, in_channels):
        super().__init__()

        self.encoder_decision = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),  # H × W sollte gleich bleiben
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # H × W bleibt gleich
            nn.ReLU()
        )

        self.decision_head = nn.Conv2d(64, 2,
                                       kernel_size=1)  # Output Shape für Decision [B, C, H, W], für jedes Grid 0 oder 1 -> C=2 [0,1]

        self.encoder_dir = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),  # H × W sollte gleich bleiben
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # H × W bleibt gleich
            nn.ReLU()
        )

        self.dir_head = nn.Conv2d(64, 4, kernel_size=1)  # C=[0-3]

    def forward(self, x):

        x_decision = self.encoder_decision(x)  # Shape bleibt (B, 64, H, W)
        decision_logits = self.decision_head(x_decision)  # → (B, 2, H, W)

        x_dir = self.encoder_dir(x)  # Shape bleibt (B, 64, H, W)
        dir_logits = self.dir_head(x_dir)  # → (B, 2, H, W)

        return decision_logits, dir_logits

def merge_actions(
    action_type_grid,  # (E, H, W) – Priorisierte Aktionsauswahl
    attack_params=None,      # (H, W, 2)
    harvest_mask=None,       # (H, W) – bool
    return_mask=None,        # (H, W) – bool
    produce_params=None,     # (H, W, 1)
    production_type=None,
    move_params=None         # (H, W, 1)
):
    """
    Erstellt einen flachen Aktionsvektor aus den Einzel-Head-Ausgaben.
    action_type_ bestimmt, welche Aktion pro Grid aktiv ist.
    Die restlichen Parameter werden gesetzt, wenn die Aktion aktiv ist.
    Format: 7 Einträge pro Zelle, wie in MicroRTS erwartet.
    """

    E ,H, W = action_type_grid.shape
    full_action = np.zeros((E, H, W, 7), dtype=np.int32)
    print("produce_params.shape:", produce_params.shape)
    for i in range(E):
        for j in range(H):
            for k in range(W):
                a_type = action_type_grid[i,j,k] #holt sich den action type
                full_action[i,j,k, 0] = a_type  # action type eintragen

                # Aktion ausführen, andere Parameterfelder auf 0
                if a_type == 5 and attack_params is not None:  #action_type=5 -> Attack
                    print("attack_params.shape:", attack_params.shape)
                    print("Beispielwert:", attack_params[i, j, k])
                    full_action[i,j,k,6]=attack_params[i,j,k]

                elif a_type == 2 and harvest_mask is not None:
                    full_action[i, j, k, 2] = harvest_mask[i, j, k]

                elif a_type == 3 and return_mask is not None:
                    full_action[i, j, k, 3] = return_mask[i, j, k]

                elif a_type == 4 and produce_params is not None and production_type is not None:
                    full_action[i, j, k, 4] = produce_params[i, j, k]
                    full_action[i, j, k, 5] = production_type[i, j, k]

                elif a_type == 1 and move_params is not None:
                    full_action[i, j, k, 1] = move_params[i, j, k]

                else:
                    # Aktion unbekannt oder nicht durchführbar
                    full_action[i,j,k, :] = 0  # no-o

    return full_action.flatten()  # (H * W * 7,)

def get_action_type_grid(attack_decision,
    harvest_decision,
    return_decision,
    produce_decision,
    move_decision):
    #print("Attack_decision_shape:", attack_decision.shape) #Attack_decision_shape: (24, 8, 8)
    E, H, W = attack_decision.shape
    print(E, H,W)
    print(attack_decision.shape)


    action_type_grid = np.full((E,H, W), 6, dtype=np.int32)
    for i in range(E):
        for j in range(H):
            for k in range(W):
                """Regelt Priorisierung der Decider
                Attack>Harvest>return>Produce>move"""
                if attack_decision[i,j,k] == 1:
                    action_type_grid[i,j,k] = 5
                elif harvest_decision[i,j,k] == 1:
                    action_type_grid[i,j,k] = 2
                elif return_decision[i,j,k] == 1:
                    action_type_grid[i,j,k] = 3
                elif produce_decision[i,j,k] == 1:
                    action_type_grid[i,j,k] = 4
                elif move_decision[i,j,k] == 1:
                    action_type_grid[i,j,k] = 1

    return action_type_grid


class Agent:
    def __init__(self, env, exp_buffer, device="cpu"):
        """
        Initialisiert den Agenten mit Zugriff auf die Umgebung und den Replay Buffer.
        """
        self.device=device
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

        self.movement_head = MovementHead(in_channels=29).to(self.device)
        self.harvest_head = MovementHead(in_channels=29).to(self.device)
        self.return_head = MovementHead(in_channels=29).to(self.device)
        self.production_head = ProduceHead(in_channels=29).to(self.device)
        self.attack_head = AttackHead(in_channels=29).to(self.device)

        self.heads = {
            "attack": self.attack_head,
            "harvest": self.harvest_head,
            "return": self.return_head,
            "move": self.movement_head,
            "produce": self.production_head,
        }

        self.head_config = {
            "attack": {"type_id": 5, "indices": (0, 6), "classes": (2, 4)},
            "harvest": {"type_id": 2, "indices": (0, 2), "classes": (2, 4)},
            "return": {"type_id": 3, "indices": (0, 3), "classes": (2, 4)},
            "move": {"type_id": 1, "indices": (0, 1), "classes": (2, 4)},
            "produce": {"type_id": 4, "indices": (0, 4, 5), "classes": (2, 4, 7)},
        }




    def _reset(self):
        """
        Startet eine neue Episode und setzt interne Zustände zurück.
        """
        self.state = self.env.reset()
        self.total_reward = 0.0



    @torch.no_grad()
    def play_step(self, epsilon=0.0, device="cpu"):
        """
        Führt einen Schritt im Environment aus:
        - Wählt eine Aktion mittels ε-greedy Strategie
        - Führt Aktion im Environment aus
        - Speichert Transition im Replay Buffer
        - Rückgabe: Gesamt-Reward bei Episodenende, sonst None
        """
        done_reward = None

        # ε-greedy Aktionsauswahl
        if np.random.random() < epsilon:
            action = np.stack([self.env.action_space.sample() for _ in range(self.env.num_envs)])
        else:
            # Zustand vorbereiten für Netzwerkeingabe
            state_a = np.array(self.state, copy=False)
            state_v = torch.tensor(self.state, dtype=torch.float32, device=self.device).permute(0, 3, 1, 2)

            """!!!Attack noch unklar über wie Direction codiert ist!!!"""
            """Der folgende Teil ist q_vals=net(state) für mehrere Köpfe. Unklar was genau net(state)"""
            """Jeder Kopf muss alle seine maximal Möglichen Aktionen machen, diese einzeln. Die beste Aktion an Merge 
            schicken, welcher die Gesamtaktion ausführt
            """
            attack_decision, attack_dir=self.attack_head(state_v)
            print("Attack Decision shape nach durchlaufen neuronales Netz:", attack_decision.shape)
            attack_mask = attack_decision.argmax(dim=1).squeeze(0).cpu().numpy()
            attack_param = attack_dir.argmax(dim=1).squeeze(0).cpu().numpy()   #attack Mask [24,8,8]
            print("Attack Decision nach durchlaufen neuronales Netz:", attack_mask.shape)


            move_decision, move_dir = self.movement_head(state_v)
            move_mask = move_decision.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W), 0 oder 1
            move_param = move_dir.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W, 1), Richtung 0–3

            harvest_decision, harvest_dir = self.harvest_head(state_v)
            harvest_mask = harvest_decision.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W), 0 oder 1
            harvest_param = harvest_dir.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W, 1), Richtung 0–3


            return_decision, return_dir = self.return_head(state_v)
            return_mask = return_decision.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W), 0 oder 1
            return_param = return_dir.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W, 1), Richtung 0–3

            """!Actung Produce und Production!"""
            production_decision, production_dir, production_type = self.production_head(state_v)
            produce_mask = production_decision.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W), 0 oder 1
            produce_param = production_dir.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W, 1), Richtung 0–3
            produce_type = production_type.argmax(dim=1).squeeze(0).cpu().numpy()

            #Führe Teilaktion zur Gesamtaktion zusammen
            action_type_grid=get_action_type_grid(attack_mask,harvest_mask, return_mask, produce_mask, move_mask)
            action=merge_actions(action_type_grid,attack_param,harvest_param,return_param, produce_type,produce_param,move_param)

            #Führe Aktion aus
        torch.tensor(self.env.venv.venv.get_action_mask(), dtype=torch.float32)
        new_state, reward, is_done, _= self.env.step(action)
        self.total_reward += reward

        exp = (self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)

        self.state = new_state
        if np.any(is_done):
            done_reward = self.total_reward
        self._reset()


        return print("Reward:", reward, "Done:", is_done)

    def calc_loss(states, actions, heads, head_config, device="cpu"):
        """
        generische, maskierte Verlustfunktion welche jeweils den zuständigen Head trainiert, abhängig davon, welcher
        action_type pro Zelle aktiv ist
        1. Für jedes Feld prüfen welcher Head zuständig ist
        2. den entsprechenden Loss für den Head berechnen wo dieser aktiv ist
        """

        total_loss = 0.0
        B, H, W, C = states.shape
        states_v = torch.tensor(states, dtype=torch.float32, device=device).permute(0, 3, 1, 2)  # [B,C,H,W]
        action_type = torch.tensor(actions[..., 0], dtype=torch.int64, device=device)  # [B,H,W]

        for name, config in head_config.items():
            head = heads[name]
            type_id = config["type_id"]
            idx_dec, *idxs_aux = config["indices"]
            classes_dec, *classes_aux = config["classes"]

            # Zielwerte
            target_dec = torch.tensor(actions[..., idx_dec], dtype=torch.long, device=device)  # [B,H,W]
            target_aux = [torch.tensor(actions[..., i], dtype=torch.long, device=device) for i in idxs_aux]

            # Head-Vorwärtsdurchlauf
            out = head(states_v)
            logits_dec = out[0].permute(0, 2, 3, 1).reshape(-1, classes_dec)  # [B*H*W, C]
            target_dec_flat = target_dec.reshape(-1)
            mask_flat = (action_type == type_id).reshape(-1)

            loss = 0.0
            if mask_flat.sum() > 0:
                loss += F.cross_entropy(logits_dec[mask_flat], target_dec_flat[mask_flat])

            for logits_aux, t_aux, c_aux in zip(out[1:], target_aux, classes_aux):
                logits_aux = logits_aux.permute(0, 2, 3, 1).reshape(-1, c_aux)
                t_aux_flat = t_aux.reshape(-1)
                loss += F.cross_entropy(logits_aux[mask_flat], t_aux_flat[mask_flat])

            total_loss += loss

        return total_loss




"""
Observation shape:  ([24, 8, 8, 29]) #[num_env, H,W, C]
move_dir.shape:     ([1, 4, 8, 8])  c=[0,3]  
move_dec.shape:     ([1, 2, 8, 8])  c=[0,1]

"""
if __name__ == "__main__":



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
        "hyperparameters", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]))
    )

    # TRY NOT TO MODIFY: seeding
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    print(f"Device: {device}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    envs = MicroRTSGridModeVecEnv(
        num_selfplay_envs=args.num_selfplay_envs,
        num_bot_envs=args.num_bot_envs,
        partial_obs=args.partial_obs,
        max_steps=2000,
        render_theme=2,
        ai2s=[microrts_ai.coacAI for _ in range(args.num_bot_envs - 6)]
        + [microrts_ai.randomBiasedAI for _ in range(min(args.num_bot_envs, 2))]
        + [microrts_ai.lightRushAI for _ in range(min(args.num_bot_envs, 2))]
        + [microrts_ai.workerRushAI for _ in range(min(args.num_bot_envs, 2))],
        map_paths=[args.train_maps[0]],
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
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


    """Start"""

    obs = envs.reset()

    envs.render(mode="human")
    _ = envs.venv.venv.get_action_mask()  # Initialisiere die source_unit_mask
    expbuffer = ExperienceBuffer(100)
    agent = Agent(envs, expbuffer, device=device)
    for i in range(10):
        test = agent.play_step(epsilon=0.5)

        print(test)

    envs.venv.venv.render(mode="human")
    """
    #teste replay Buffer
    for _ in range(20):  # ein paar Schritte generieren
        agent.play_step( epsilon=0.1, device=device)

    test_replay_buffer_once(expbuffer, expected_shape=(10, 10, 7), batch_size=4)
    """

    input("Drücke Enter, um die Umgebung zu schließen...")
    envs.close()
