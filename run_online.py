from collections import deque
from functools import partial
from itertools import chain

import config
import doy
import env_utils
import paths
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from doy import PiecewiseLinearSchedule as PLS
from torch.distributions import Categorical
from torch.utils.data import DataLoader, TensorDataset
from utils import create_decoder
import numpy as np
import data_loader

from data_loader import normalize_obs

from replay_buffer_tensordict import ReplayBuffer

import datetime

device_copy = config.DEVICE
print(device_copy)


cfg = config.get()
current_datetime = datetime.datetime.now()
formatted_datetime_1 = current_datetime.strftime("%Y%m%d")
formatted_datetime_2 = current_datetime.strftime("%H%M%S")
cfg.stage_exp_name = formatted_datetime_1+'_'+formatted_datetime_2 + '_online-ft'

doy.print("[bold green]BCV-LR: online finetuning and policy imitation")
config.print_cfg(cfg)

run, logger = config.wandb_init("online_ft_results", config.get_wandb_cfg(cfg),wandb_enabled = True)

state_dict = torch.load(paths.get_latent_policy_path(config.get().exp_name))



train_data, test_data = data_loader.load(cfg.env_name)
print(train_data)
train_iter = train_data.get_iter(cfg.offline_pt.bs)
test_iter = test_data.get_iter(128)


policy = utils.create_policy(
    cfg.model,
    action_dim=cfg.model.la_dim,
    state_dict=state_dict["policy"],
    strict_loading=True,
)

policy_decoder = create_decoder(
    in_dim=cfg.model.la_dim,
    out_dim=cfg.model.ta_dim,
    hidden_sizes=(192, 128, 64),
)

idm_decoder = create_decoder(
    in_dim=cfg.model.la_dim,
    out_dim=cfg.model.ta_dim,
    hidden_sizes=(192, 128, 64),
)


models_path = paths.get_models_path(cfg.exp_name)
idm, wm = utils.create_dynamics_models(cfg.model, state_dicts=torch.load(models_path))

policy.train()
idm.train()
policy_decoder.train()
idm_decoder.train()




envs = env_utils.setup_procgen_env(
    num_envs=cfg.online_ft.num_envs,
    env_id=cfg.env_name,
    gamma=cfg.online_ft.gamma,
)




idm_opt_idm = torch.optim.Adam(idm.parameters(),lr= cfg.ta_idm_lr)
idm_opt_decoder = torch.optim.Adam(idm_decoder.parameters())

policy_opt = torch.optim.Adam(utils.chain(policy.parameters()),lr = cfg.latent_policy_imitaiton.lr)



replay_buffer_capacity = cfg.buffer_size
replay_buffer = ReplayBuffer(max_size = replay_buffer_capacity, ta_dim = 15)


buf_obs = deque(maxlen=3)




def action_selection_hook_onlysl(next_obs: torch.Tensor, global_step: int = None, action=None):
    # sample action
    with torch.no_grad():
        logits_sl = idm_decoder(policy(next_obs))
    logits = logits_sl
    probs = Categorical(logits=logits)

    action_given = action is not None

    if not action_given:
        action = probs.sample()

    
    
    

    return action, probs.log_prob(action), probs.entropy()

def random_selection_hook(next_obs: torch.Tensor, global_step: int = None, action=None):
    # sample action
    actions = np.random.randint(0, 15, size = (cfg.online_ft.num_envs,), dtype='int32')

    

    return actions


def reset_decoder(decoder):
    for layer in decoder.children():
        if isinstance(layer, torch.nn.Linear):
            layer.reset_parameters()
        else:
            assert isinstance(layer, torch.nn.ReLU)



def finetuning_policylearning_each_iteration(update, global_step):
    
    
    idm.train()
    idm_decoder.train()

    
    

    collect_iter = replay_buffer.get_iter(batch_size=512,device = config.DEVICE)

    
    
    times = cfg.idm_times
    count = 0
    while count < times:
        batch = next(collect_iter)
        inputs = batch['obs']
        labels = batch['ta']
        
        idm_opt_idm.zero_grad()
        idm_opt_decoder.zero_grad()
        outputs = idm_decoder(idm(inputs)[0]["la"])
        loss = F.cross_entropy(outputs, labels)

        wm_loss, vq_perp = wm_collect_loss(batch)

        loss = loss + cfg.wm_loss_alpha*wm_loss
        
        loss.backward()
        idm_opt_idm.step()
        idm_opt_decoder.step()
        count+=1
        if count == times:
            break

    logger(
            step=global_step,
            idm_supervised_loss = loss.item(),
            wm_unsupervised_loss = wm_loss.item(),
            vq_perp = vq_perp,
            update_times = times
        )

    idm.eval()
    idm_decoder.eval()
    _, eval_metrics = utils.eval_latent_repr_ta(test_data, idm, idm_decoder)

    logger(global_step, **eval_metrics)

  
    
    times = cfg.policy_times
    count = 0
    while count < times:
        batch = next(train_iter)
        with torch.no_grad():
            idm.label(batch)

        preds = policy(batch["obs"][:, -2])
        loss = F.mse_loss(preds, batch["la"])

        policy_opt.zero_grad()
        loss.backward()
        policy_opt.step()
        count +=1

        
    
    logger(
            step=global_step,
            bc_loss = loss.item(),
        )
    

        
    torch.cuda.empty_cache()
    #buf_obs.clear()
    #buf_la.clear()
    #buf_ta.clear()

def update_wm_collect(wm_opt, batch_collect):
    idm.train()
    wm.train()
    #batch = next(train_iter)

    #vq_loss, vq_perp = idm.label(batch)
    #wm_loss = wm.label(batch)

    vq_loss_collect, vq_perp_collect = idm.label(batch_collect)
    wm_loss_collect = wm.label(batch_collect)


    loss = vq_loss_collect + wm_loss_collect

    wm_opt.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_([*idm.parameters(), *wm.parameters()], 2)
    wm_opt.step()

def wm_collect_loss(batch_collect):
    idm.train()
    wm.train()
    #batch = next(train_iter)

    #vq_loss, vq_perp = idm.label(batch)
    #wm_loss = wm.label(batch)

    vq_loss_collect, vq_perp_collect = idm.label(batch_collect)
    wm_loss_collect = wm.label(batch_collect)


    loss = vq_loss_collect + wm_loss_collect

    return loss, vq_perp_collect




def merge_TC_dims(x: torch.Tensor):
    """x.shape == (B, T, C, H, W) -> (B, T*C, H, W)"""
    return x.reshape(x.shape[0], -1, *x.shape[3:])

def add_replaybuffer_batch(next_obs, actions):
    buf_obs.append(next_obs.unsqueeze(1))   # 16,1,3,64,64
    if len(buf_obs) == 3:
        #buf_la.append(idm(torch.cat(list(buf_obs), dim=1))[0]["la"])
        #buf_obs_list.append(torch.cat(list(buf_obs), dim=1))
        obs_stack = torch.cat(list(buf_obs), dim=1)
        replay_buffer.add(obs_stack.cpu().numpy(), actions)
    

    
def train():

    online_cfg = cfg.online_ft
    device = config.DEVICE
    global_step = 0
    steps_train = int(150000 / (online_cfg.num_steps*online_cfg.num_envs))    # results are obtained at 100k instead of 150k 

    next_obs = torch.from_numpy(envs.reset()).permute((0, 3, 1, 2)).to(device)
    next_done = torch.zeros(online_cfg.num_envs).to(device).float()

    for step in doy.loop(1, 65 , desc="seed_step"):
        with torch.no_grad():
            actions= random_selection_hook(normalize_obs(next_obs), global_step)
        next_obs, reward, done, info = envs.step(actions)
        
        next_obs = torch.from_numpy(next_obs).permute((0, 3, 1, 2)).to(device)
        next_done = torch.from_numpy(done).to(device).float()

        add_replaybuffer_batch(next_obs, actions)

    

    for update in doy.loop(1, steps_train + 1, desc="Online finetuning and policy imitation"):
        
        finetuning_policylearning_each_iteration(update, global_step)


        for step in range(0, online_cfg.num_steps):
            #print(111)
            

            # [acting]
            with torch.no_grad():
                if True:
                    actions, _, _ = action_selection_hook_onlysl(
                        normalize_obs(next_obs), global_step
                    )
                


            # [env.step]
            next_obs, reward, done, info = envs.step(actions.cpu().numpy())  # reward not used in BCV-LR
            next_obs = torch.from_numpy(next_obs).permute((0, 3, 1, 2)).to(device)
            next_done = torch.from_numpy(done).to(device).float()

            add_replaybuffer_batch(next_obs, actions.cpu().numpy())

            for substep, item in enumerate(info):
                if "episode" in item.keys():
                    logger(
                        step=global_step + substep,
                        global_step=global_step + substep,
                        episodic_return=item["episode"]["r"],
                        episodic_length=item["episode"]["l"],
                    )
                    break


            global_step += 1 * online_cfg.num_envs
    


train()
