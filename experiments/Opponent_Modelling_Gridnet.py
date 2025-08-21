import argparse
from distutils.util import strtobool
import time
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor, VecVideoRecorder
from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
from collections import deque
import os

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
    parser.add_argument('--eval_interval', type=int, default=10000,
                        help='Wie häufig evaluiert wird')

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

class StatePredictionNet(nn.Module):
    def __init__(self, input_shape, hidden_channels=64):
        super(StatePredictionNet, self).__init__()
        self.input_shape = input_shape  # (c, h, w)
        c, h, w = input_shape
        print(c,h,w)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(c, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Bottleneck (optional)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1),  # Output-Kanal = 1
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: Tensor der Form [batch_size, c, h, w]
        Rückgabe: Tensor der Form [batch_size, c, h, w]
        """
        z = self.encoder(x)
        z = self.bottleneck(z)
        out = self.decoder(z)

        print("Predicted max:", out.max().item(), "min:", out.min().item())
        return out

import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque

class StateModeling:
    def __init__(self, env, net, device, buffer_size=10000, batch_size=32, agent=None):
        self.env = env
        self.net = net.to(device)
        self.device = device
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.agent = agent
        self.state = self.env.reset()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([15.0]).to(self.device))
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)

    def train(self):
        # Schritt 1: Sammle Daten
        num_envs=self.env.num_envs
        states = []
        for _ in range(self.batch_size + 10):  # +10 für Vorhersage von t+10

            states.append(self.state)
            self.state, _, _, _ = self.agent.play_step(epsilon=0.5)

        # Schritt 2: Training vorbereiten
        total_loss = 0.0
        self.optimizer.zero_grad()

        for i in range(self.batch_size):
            for env_i in range(num_envs):
                input_state = torch.tensor(states[i][env_i], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)
                target_state = torch.tensor(states[i + 10][env_i], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)

                predicted_state = self.net(input_state)

                target_channel12 = target_state[:, 12:13, :, :]  # Shape [1, 1, H, W]
                loss = self.loss_fn(predicted_state, target_channel12)
                loss.backward()
                total_loss += loss.item()

        self.optimizer.step()
        return total_loss / (self.batch_size * num_envs)

    def evaluat(self):
        num_envs = self.env.num_envs
        states = []
        num_units = 0
        num_predUnits = 0
        num_correctUnits = 0

        for _ in range(100 + 10):  # +10 für Vorhersage von t+10
            states.append(self.state)
            self.state, _, _, _ = self.agent.play_step(epsilon=0.5)

        for i in range(100):
            for env_i in range(num_envs):
                input_state = torch.tensor(states[i][env_i], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(
                    self.device)
                target_state = torch.tensor(states[i + 10][env_i], dtype=torch.float32).permute(2, 0, 1).unsqueeze(
                    0).to(self.device)
                predicted_state = self.net(input_state)

                for h in range(8):
                    for w in range(8):
                        is_true_unit = target_state[0, 12, h, w] > 0.5
                        is_pred_unit = predicted_state[0, 0, h, w] > 0.5  # Index 0, da nur 1 Kanal!

                        if is_true_unit and is_pred_unit:
                            num_correctUnits += 1
                        if is_true_unit:
                            num_units += 1
                        if is_pred_unit:
                            num_predUnits += 1

        return num_units, num_predUnits, num_correctUnits



class RandomAgent:
    def __init__(self, env):
        self.env = env
        self.num_envs = env.num_envs


    def predict(self, obs=None, deterministic=False):
        # Gibt zufällige Aktionen für alle Envs zurück
        if obs is None:
            obs = self.last_obs
        actions = self.env.action_space.sample()
        return actions, None

    def play_step(self, epsilon=0.0):
        _= self.env.venv.venv.get_action_mask()
        actions = self.env.action_space.sample()
        next_obs, rewards, dones, infos = self.env.step(actions)
        self.last_obs = next_obs
        return next_obs, rewards, dones, infos

    def get_env(self):
        return self.env
if __name__ == "__main__":

    args = parse_args()
    print(args.exp_name)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print("PID: ", os.getpid())
    print(f"Device: {device}")
    num_envs = args.num_bot_envs
    num_each = num_envs // 4
    reward_weights = np.array([50.0, 3.0, 3.0, 0.0, 5.0, 1.0])
    env = MicroRTSGridModeVecEnv(
        num_selfplay_envs=args.num_selfplay_envs,
        num_bot_envs=args.num_bot_envs,
        partial_obs=args.partial_obs,
        max_steps=args.max_steps,
        render_theme=2,
        ai2s=(
            [microrts_ai.workerRushAI for _ in range(args.num_bot_envs)]+
            [microrts_ai.workerRushAI for _ in range(num_each)] +
            [microrts_ai.lightRushAI for _ in range(num_each)] +
            [microrts_ai.coacAI for _ in range(num_envs - 3 * num_each)]
        ),
        map_paths=[args.train_maps[0]],
        reward_weight=reward_weights,
        cycle_maps=args.train_maps,
    )
    env = MicroRTSStatsRecorder(env, args.gamma)
    env = VecMonitor(env)

    print(type(env))  # Welcher VecEnv-Typ?
    print(env.reset().shape)  # Format prüfen
    print(env.observation_space.shape)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_shape = env.observation_space.shape  # (H, W, C)
    input_shape = (obs_shape[2], obs_shape[0], obs_shape[1])  # (C, H, W)
    random_agent = RandomAgent(env)
    model = StatePredictionNet(input_shape=input_shape)
    state_modeling = StateModeling(env=env, net=model, agent=random_agent, device=device)

    for epoch in range(1000):
        avg_loss = state_modeling.train()
        print(f"Epoch {epoch}: avg loss = {avg_loss}")
        if epoch % 100 ==0:
            num_units, num_predUnits, num_correctUnits = state_modeling.evaluat()
            print(f"num_units, num_predUnits, num_correctUnits: {num_units, num_predUnits, num_correctUnits}")

