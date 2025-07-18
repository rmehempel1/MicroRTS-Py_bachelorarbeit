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

        # ⚠️ +2 für (x, y) Position
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size + 2, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

        self.heads = nn.ModuleList([
            nn.Linear(512, 6),   # action type
            nn.Linear(512, 4),   # move_dir
            nn.Linear(512, 4),   # harvest_dir
            nn.Linear(512, 4),   # return_dir
            nn.Linear(512, 7),   # produce_type
            nn.Linear(512, 49),  # attack_dir
        ])

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
        out = [head(fc_out) for head in self.heads]
        return out

    def _get_conv_out(self, shape):
        o = self.encoder(torch.zeros(1, *shape))
        return int(np.prod(o.size()))


class ReplayBuffer:
    def __init__(self, capacity, state_shape, action_shape):
        self.buffer = deque(maxlen=capacity)
        self.state_shape = state_shape
        self.action_shape = action_shape

    def append(self, state, action, reward, done, next_state, unit_pos):
        self.buffer.append((state, action, reward, done, next_state, unit_pos))

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        samples = [self.buffer[idx] for idx in indices]

        states, actions, rewards, dones, next_states, unit_positions = zip(*samples)

        states = np.array(states, copy=False)  # [B, H, W, C]
        actions = np.array(actions, copy=False)  # [B, 7]
        rewards = np.array(rewards, copy=False)  # [B]
        dones = np.array(dones, copy=False)  # [B]
        next_states = np.array(next_states, copy=False)  # [B, H, W, C]
        unit_positions = np.array(unit_positions, copy=False)  # [B, 2]

        return states, actions, rewards, dones, next_states, unit_positions

class Agent:
    def __init__(self, env, exp_buffer, net, device="cpu"):
        self.env=env
        self.device=device
        self.exp_buffer = exp_buffer
        self._reset()
        self.net = net
        self.state = self.env.reset()
        self.total_reward=0.0

    def _reset(self):
        """
        Startet eine neue Episode und setzt interne Zustände zurück.
        """
        self.state = self.env.reset()
        self.total_reward = 0.0


    @torch.no_grad()
    def play_step(self, epsilon=0.0):
        net = self.net
        device = self.device
        raw_masks = self.env.venv.venv.get_action_mask()  # [num_envs, H*W, 78]
        e, h, w, _ = self.state.shape
        grid_size = int(np.sqrt(raw_masks.shape[1]))
        flat_index = h * grid_size + w



        if np.random.random()< epsilon:
            """aus Gridnet kopiert, zufällige Aktion"""

            grid_size = int(np.sqrt(raw_masks.shape[1]))

            def sample_valid(mask_2d):
                valid_indices = np.where(mask_2d)[0]
                if len(valid_indices) == 0:
                    return 0  # Fallback auf 0
                return np.random.choice(valid_indices)

            full_action = np.zeros((self.env.num_envs, grid_size, grid_size, 7), dtype=np.int32)
            full_action_taken_grid = np.zeros((self.env.num_envs, grid_size, grid_size), dtype=np.int32)
            for env_i in range(self.env.num_envs):
                for idx in range(grid_size * grid_size):
                    cell_mask = raw_masks[env_i, idx]
                    i, j = divmod(idx, grid_size)

                    a_type = sample_valid(cell_mask[0:6])
                    full_action[env_i, i, j, 0] = a_type
                    full_action_taken_grid[env_i, i, j] = a_type
                    if a_type == 1:  # Move
                        full_action[env_i, i, j, 1] = sample_valid(cell_mask[6:10])
                    elif a_type == 2:  # Harvest
                        full_action[env_i, i, j, 2] = sample_valid(cell_mask[10:14])
                    elif a_type == 3:  # Return
                        full_action[env_i, i, j, 3] = sample_valid(cell_mask[14:18])
                    elif a_type == 4:  # Produce
                        full_action[env_i, i, j, 4] = sample_valid(cell_mask[18:22])
                        full_action[env_i, i, j, 5] = sample_valid(cell_mask[22:29])
                    elif a_type == 5:  # Attack
                        full_action[env_i, i, j, 6] = sample_valid(cell_mask[29:78])

            full_action = full_action.reshape(self.env.num_envs, -1)
            # Führe Aktion aus
            torch.tensor(self.env.venv.venv.get_action_mask(), dtype=torch.float32)
            new_state, reward, is_done, infos = self.env.step(full_action)
            self.total_reward += reward
        else:
            full_action=np.zeros((h, w, 7), dtype=np.int32)


            for env_i in range(e):
                for i in range(h):
                    for j in range(w):
                        if self.state[env_i,i,j,10]==1 and self.state[env_i,i,j,21]==0: #eiugen Einheit und keine Action

                            state_a = np.array([self.state], copy=False)  # [1, H, W, C]
                            state_a = np.transpose(state_a, (0, 3, 1, 2))  # ➜ [1, C, H, W]
                            state_v = torch.tensor(state_a, dtype=torch.float32, device=device)
                            unit_pos = torch.tensor([[j, i]], dtype=torch.float32, device=device)  # [B, 2]
                            q_vals_v = net(state_v, unit_pos=unit_pos)

                            # Maske für action_type (NONE, MOVE, ..., ATTACK) auf Zelle [env_i,i,j]
                            mask = torch.tensor(raw_masks[e, flat_index, 0:6], dtype=torch.bool,device=q_vals_v[0].device)
                            logits = q_vals_v[0][env_i,i,j]                       # Q-Werte der Zelle extrahieren
                            masked_logits = logits.masked_fill(~mask, -1e9)     # Ungültige Werte maskieren
                            action_type = torch.argmax(masked_logits).item()    # Beste gültige Aktion auswählen
                            full_action[env_i,i,j, 0] = action_type               # In Aktionsarray eintragen

                            # Nachfolgende Heads abhängig vom Action-Type
                            if action_type == 0:  # NONE
                                pass

                            elif action_type == 1:  # MOVE (6:10)
                                mask = torch.tensor(raw_masks[e, flat_index, 6:10], dtype=torch.bool,
                                                    device=q_vals_v[1].device)
                                logits = q_vals_v[1][env_i,i,j]
                                masked_logits = logits.masked_fill(~mask, -1e9)
                                move_dir = torch.argmax(masked_logits).item()
                                full_action[env_i,i,j, 1] = move_dir

                            elif action_type == 2:  # HARVEST (10:14)
                                mask = torch.tensor(raw_masks[e, flat_index, 10:14], dtype=torch.bool,
                                                    device=q_vals_v[2].device)
                                logits = q_vals_v[2][env_i,i,j]
                                masked_logits = logits.masked_fill(~mask, -1e9)
                                harvest_dir = torch.argmax(masked_logits).item()
                                full_action[env_i,i,j, 2] = harvest_dir

                            elif action_type == 3:  # RETURN (14:18)
                                mask = torch.tensor(raw_masks[e, flat_index, 14:18], dtype=torch.bool,
                                                    device=q_vals_v[3].device)
                                logits = q_vals_v[3][env_i,i,j]
                                masked_logits = logits.masked_fill(~mask, -1e9)
                                return_dir = torch.argmax(masked_logits).item()
                                full_action[env_i,i,j, 3] = return_dir

                            elif action_type == 4:  # PRODUCE
                                # Direction (18:22)
                                mask = torch.tensor(raw_masks[e, flat_index, 18:22], dtype=torch.bool,
                                                    device=q_vals_v[4].device)
                                logits = q_vals_v[4][env_i,i,j]
                                masked_logits = logits.masked_fill(~mask, -1e9)
                                produce_dir = torch.argmax(masked_logits).item()
                                full_action[env_i,i,j, 4] = produce_dir

                                # Unit-Type (22:29)
                                mask = torch.tensor(raw_masks[e, flat_index, 22:29], dtype=torch.bool,
                                                    device=q_vals_v[5].device)
                                logits = q_vals_v[5][env_i,i,j]
                                masked_logits = logits.masked_fill(~mask, -1e9)
                                produce_type = torch.argmax(masked_logits).item()
                                full_action[env_i,i,j, 5] = produce_type

                            elif action_type == 5:  # ATTACK (29:78)
                                mask = torch.tensor(raw_masks[e, flat_index, 29:78], dtype=torch.bool,
                                                    device=q_vals_v[6].device)
                                logits = q_vals_v[6][env_i,i,j]
                                masked_logits = logits.masked_fill(~mask, -1e9)
                                attack_target = torch.argmax(masked_logits).item()
                                full_action[env_i,i,j, 6] = attack_target

            torch.tensor(self.env.venv.venv.get_action_mask(), dtype=torch.float32)
            new_state, reward, is_done, infos = self.env.step(full_action)

        self.total_reward += reward
        for i in range(self.env.num_envs):
            for i in range(h):
                for j in range(w):
                    if self.state[env_i, i, j, 10] == 1 and self.state[env_i, i, j, 21] == 0:
                        single_action = full_action[env_i, i, j]  # [7]
                        self.exp_buffer.append(
                            self.state[env_i],  # [H, W, C]
                            single_action,  # [7]
                            reward[env_i],  # float
                            is_done[env_i],  # bool
                            new_state[env_i],  # [H, W, C]
                            (j, i)  # unit_pos (x, y)
                        )


        self.state = new_state
        if np.any(is_done):
            done_reward = self.total_reward
            self._reset()
            return {"done": True, "reward": done_reward[0], "infos": infos[0]}
        return {"done": False, "reward": reward[0], "infos": infos[0]}

    def calc_loss(self, batch, tgt_net, gamma):
        states, actions, rewards, dones, next_states, unit_positions = batch

        device = self.device
        B = len(states)

        # [B, C, H, W]
        states_v = torch.tensor(states, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
        next_states_v = torch.tensor(next_states, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
        actions_v = torch.tensor(actions, dtype=torch.int64, device=device)  # [B, 7]
        rewards_v = torch.tensor(rewards, dtype=torch.float32, device=device)  # [B]
        done_mask = torch.tensor(dones, dtype=torch.bool, device=device)  # [B]
        unit_pos = torch.tensor(unit_positions, dtype=torch.long, device=device)  # [B, 2] (x, y)

        # Q-Werte nur für relevante Einheiten berechnen
        qvals = self.net(states_v, unit_pos=unit_pos)  # Liste: [B, num_actions] je Head
        qvals_next = tgt_net(next_states_v, unit_pos=unit_pos)

        total_loss = 0.0
        for head_idx in range(len(qvals)):
            qval = qvals[head_idx].gather(1, actions_v[:, head_idx].unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_qval = qvals_next[head_idx].max(1).values
                next_qval[done_mask] = 0.0
                q_target = rewards_v + gamma * next_qval

            loss = F.smooth_l1_loss(qval, q_target)
            total_loss += loss

        return total_loss


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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    """
    ([microrts_ai.passiveAI for _ in range(args.num_bot_envs // 2)] +
          [microrts_ai.workerRushAI for _ in range(args.num_bot_envs // 2)]),
    """
    reward_weights = np.array([100.0, 1.0, 10.0, -100.0, 10.0, -100.0])
    print("Reward Weights:", reward_weights)
    envs = MicroRTSGridModeVecEnv(
        num_selfplay_envs=args.num_selfplay_envs,
        num_bot_envs=args.num_bot_envs,
        partial_obs=args.partial_obs,
        max_steps=args.max_steps,
        render_theme=2,
        ai2s= [microrts_ai.passiveAI for _ in range(args.num_bot_envs)],

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
    target_net.load_state_dict(policy_net.state_dict())  # 🟢 Initiales Sync
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
    warmup_frames = 100000
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
    target_net.load_state_dict(policy_net.state_dict())  # 🟢 Initiales Sync
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
            epsilon = 1.0

        eval_reward = 0.0
        if frame_idx % eval_interval == 0:
            # eval_reward = evaluate(agent, eval_env, device=device)
            print(f"[EVAL] Frame {frame_idx} Durchschnittlicher Reward: {eval_reward:.2f}")

        # Schritt ausführen
        step_info = agent.play_step(epsilon=epsilon)
        done = step_info["done"]
        reward = step_info["reward"]
        infos = step_info["infos"]
        raw_rewards = infos.get("raw_rewards", None)

        for name, value in zip(reward_names, raw_rewards):
            reward_counts[name] += value

        if len(expbuffer) < args.batch_size:
            continue

        # Target-Sync
        if frame_idx % args.sync_interval == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Checkpoint speichern
        if frame_idx % 300000 == 0:
            torch.save(policy_net.state_dict(), f"checkpoints/{args.exp_name}_policy_{frame_idx}.pth")

        # Training
        batch = expbuffer.sample(args.batch_size)
        optimizer.zero_grad()
        loss = agent.calc_loss(batch, target_net, gamma=args.gamma)
        loss.backward()
        optimizer.step()

        # Logging
        if done:
            episode_idx += 1
            reward_queue.append(reward)
            mean_reward = np.mean(reward_queue)
            dauer = frame_idx - frame_start
            frame_start = frame_idx

            print(f"Episode: {episode_idx} Frame: {frame_idx} "
                  f"Reward: {reward:.2f} Mean Reward: {mean_reward:.2f} Eval Reward: {eval_reward:.2f} "
                  f"Loss: {loss:.4f} Epsilon: {epsilon:.4f} Dauer: {dauer}")

            log_episode_to_csv(
                csv_path=csv_path,
                episode_idx=episode_idx,
                frame_idx=frame_idx,
                reward=reward,
                mean_reward=mean_reward,
                eval_reward=eval_reward,
                loss=loss.item(),
                epsilon=epsilon,
                dauer=dauer,
                reward_counts=reward_counts,
                reward_names=reward_names
            )

            for name in reward_names:
                print(f"{name}: {reward_counts[name]}")
                reward_counts[name] = 0

            # Bestes Modell speichern
            if frame_idx > warmup_frames and (best_mean_reward is None or mean_reward > best_mean_reward):
                print(
                    f"Neues bestes Ergebnis: old mean reward: {best_mean_reward:.2f}" if best_mean_reward is not None else "old mean reward: None",
                    f"new: {mean_reward:.2f}" if mean_reward is not None else "new: None")

                best_mean_reward = mean_reward
                torch.save(policy_net.state_dict(), os.path.join(model_dir, f"{args.exp_name}_best.pth"))

    # Training fertig – final speichern
    torch.save(policy_net.state_dict(), os.path.join(model_dir, f"{args.exp_name}_final.pth"))
    print("Training abgeschlossen.")

    envs.close()



