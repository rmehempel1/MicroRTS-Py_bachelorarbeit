
import argparse
import os
import random
import subprocess
import time
from distutils.util import strtobool
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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

def sample_valid_action(mask, action_space):
    """
    Erzeugt eine gültige Aktion unter Berücksichtigung der Maske.
    mask: Shape (64, total_action_components)
    action_space: gym.spaces.MultiDiscrete([...]) z. B. [6, 4, 4, 4, 4, 4, 4]
    erstelt von gpt
    """
    nvec = action_space.nvec
    total_units = mask.shape[0]  # = 64
    total_components = len(nvec)
    action = np.zeros((total_units, total_components), dtype=np.int32)

    start = 0
    for comp_idx, choices in enumerate(nvec):
        end = start + choices
        for unit in range(total_units):
            valid_choices = np.flatnonzero(mask[unit, start:end])
            if len(valid_choices) > 0:
                action[unit, comp_idx] = np.random.choice(valid_choices)
            else:
                action[unit, comp_idx] = 0  # default fallback
        start = end
    return action

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
    # print("Initial Observation:", obs) # (24, 7 * 256)
    # envs.venv.venv.render(mode="human") # rendert das Spiel

    print("Observation shape:", obs.shape)
    # (24, 8, 8, 29) (num_envs, height, width, Features per Tile)
    print("feindlicher Worker   ", obs[1, 1, 1])
    print("Eigene Base:         ", obs[1, 6, 5])
    print("Eigener Worker       ", obs[1, 6, 6])
    print("Ressource:           ", obs[1, 7, 7])
    print("Leeres Feld:         ", obs[1, 4, 4])

    """
    #zufällige Aktionen
    for _ in range(1000):
        full_mask = envs.venv.venv.get_action_mask()  # (num_envs, 64, 22)

        all_actions = []
        for env_idx in range(envs.num_envs):
            valid_action = sample_valid_action(
                full_mask[env_idx], envs.action_space
            )  # Shape: (64, 7)
            all_actions.append(valid_action)

        actions = np.array(all_actions).reshape(envs.num_envs, -1)  # z. B. (24, 448)
        obs, _, _, _ = envs.step(actions)
        envs.render(mode="human")
    """


    def random_actions():

        full_mask = torch.tensor(envs.venv.venv.get_action_mask(), dtype=torch.float32)
        print("Action dims:", envs.action_plane_space.nvec.tolist())
        print("Summe:", sum(envs.action_plane_space.nvec.tolist()))
        print("Full mask shape:", full_mask.shape)
        action_dims = envs.action_plane_space.nvec.tolist()  # z. B. [6, 4, 2, ...]
        batch_size = full_mask.shape[0]

        # Splitte mask entlang Actiontypen (wie in PPO)
        split_masks = torch.split(full_mask, action_dims, dim=1)

        all_actions = []
        for mask in split_masks:
            # mask: shape (B, A_i) → bool
            # Setze ungültige auf 0, gültige auf 1
            valid_mask = mask.bool()

            # Zufallslogits und maskiertes Sampling
            random_logits = torch.rand_like(mask)  # Zufallswerte
            random_logits[~valid_mask] = -1e10  # Ungültige Aktionen maskieren
            action = torch.argmax(random_logits, dim=1)
            all_actions.append(action)

        # Stack zu (B, num_action_types)
        actions = torch.stack(all_actions, dim=1).cpu().numpy()
        obs, _,_,_ =envs.step(actions)


    """
    def make_move_action(y, x, direction):
        action = np.zeros((8, 8, 7), dtype=np.int32)
        action[y, x, 0] = 1  # action_type: MOVE
        action[y, x, 1] = direction  # direction: 0 = NORTH, 1 = EAST, etc.
        action = action.reshape(-1)
        action = np.stack([action] * 24)
        obs, rs, ds, infos = envs.step(action)

        return action


    def harvest_action(y, x, direction):
        action = np.zeros((8, 8, 7), dtype=np.int32)
        action[y, x, 0] = 2  # action_type: Harvest
        action[y, x, 1] = direction  # direction: 0 = NORTH, 1 = EAST, etc.
        action = action.reshape(-1)
        action = np.stack([action] * 24)
        next_obs, rs, ds, infos = envs.step(action)
        for _ in range(10):  # um den cooldown abzuwarten
            next_obs, rs, ds, infos = envs.step(action)
        return action


    def return_action(y, x, direction):
        action = np.zeros((8, 8, 7), dtype=np.int32)
        action[y, x, 0] = 3  # action_type: return
        action[y, x, 1] = direction  # direction: 0 = NORTH, 1 = EAST, etc.
        action = action.reshape(-1)
        action = np.stack([action] * 24)
        next_obs, rs, ds, infos = envs.step(action)
        for _ in range(8):  # um den cooldown abzuwarten
            next_obs, rs, ds, infos = envs.step(action)
        return action

    """


    def get_action(y, x, action_type, action_direction):
        action = np.zeros((8, 8, 7), dtype=np.int32)

        # ActionType setzen
        action[y, x, 0] = action_type

        # Richtige Parameter-Ebene wählen
        if action_type == 1:
            action[y, x, 1] = action_direction  # move
        elif action_type == 2:
            action[y, x, 2] = action_direction  # harvest
        elif action_type == 3:
            action[y, x, 3] = action_direction  # return
        elif action_type == 5:
            action[y, x, 6] = action_direction  # attack

        action = np.stack([action] * envs.num_envs)

        full_mask = envs.venv.venv.get_action_mask()
        """
        index = 9
        flat_attack_index = 20 + action_direction  # Beispiel: attack beginnt bei 20
        mask = full_mask[0][index]

        print(f"[DEBUG] Mask for unit ({y},{x}):", mask)
        print(f"[DEBUG] Attack index:", flat_attack_index)

        if mask[flat_attack_index] == 0:
            print(f"[WARNUNG] Ungültiger Angriff in Richtung {action_direction}")
            return
        """
        # Aktion ausführen
        envs.step(action.reshape(-1))

        # Leere Schritte zur Animation
        noop = np.zeros_like(action)
        noop = np.stack([noop] * envs.num_envs)

        timer = 20 if action_type == 2 else 20
        for _ in range(timer):
            envs.step(noop.reshape(-1))
            envs.venv.venv.render(mode="human")

        time.sleep(0.2)


    def get_production(y,x,production_type, production_direction):

        action = np.zeros((8,8,7), dtype=np.int32)
        action[y,x,0]=4
        action[y,x,4]=production_direction
        action[y,x,5]=production_type
        action=np.stack([action]*envs.num_envs)
        full_mask = envs.venv.venv.get_action_mask()
        next_obs,_,_,_ = envs.step(action.reshape(-1))
        action = np.zeros((8, 8, 7), dtype=np.int32)
        action=np.stack([action]*envs.num_envs)
        for _ in range(200):
            next_obs, _, _, _ = envs.step(action.reshape(-1))
            envs.venv.venv.render(mode="human")

        time.sleep(0.2)





    """
        for _ in range(3):
        step=get_action(6,6,1,2)
        step=get_action(7, 6, 2, 1)
        step = get_action(7, 6, 1, 0)
        step = get_action(6, 6, 3, 3)
    step = get_production(6, 5, 3, 0)
    step = get_action(5,5,1,0)
    step = get_action(4,5,1,0)
    step = get_action(3,5,1,0)
    step =get_action(2,5,1,3)
    step = get_action(2,4,1,3)
    step = get_action(2,3,1,3)
    step = get_action(2, 2, 1, 3)
    step = get_production(2,1,2,3)
    step=get_production(2,0,4,0)
    full_mask = envs.venv.venv.get_action_mask()
    action_direction=1
    index = 9
    flat_attack_index = 20 + action_direction  # Beispiel: attack beginnt bei 20
    mask = full_mask[0][index]

    print(f"[DEBUG] Mask for unit ({1},{0}):", mask)
    print(f"[DEBUG] Attack index:", flat_attack_index)

   
    step=get_action(1,0,5,1)
    step = get_action(1, 0, 5, 2)
    step = get_action(1, 0, 5, 3)
    step = get_action(1, 0, 5, 4)




    step = get_action(2,2,1,3)
    step = get_action(2,1,5,0)
    step = get_action(2, 1, 5, 0)
    step = get_action(2, 1, 5, 1)
    step = get_action(2, 1, 5, 2) 
    step = get_action(2, 1, 5, 3)
    

    envs.venv.venv.render(mode="human")

    """
    print("Observation Shape:", envs.observation_space.shape)
    for _ in range(1000):
        random_actions()


    input("Drücke Enter, um die Umgebung zu schließen...")
    envs.close()
