from tensordict import TensorDict, TensorDictBase
import torch
from torch.utils.data import DataLoader
from collections.abc import Generator


def _unfold_td(td: TensorDictBase, seq_len: int, unfold_step: int = 1):
    """
    Unfolds the given TensorDict along the time dimension.
    The unfolded TensorDict shares its underlying storage with the original TensorDict.
    """
    res_batch_size = (td.batch_size[0] - seq_len + 1,)
    td = td.apply(
        lambda x: x.unfold(0, seq_len, unfold_step).movedim(-1, 1),
        batch_size=res_batch_size,
    )
    return td


def normalize_obs(obs: torch.Tensor) -> torch.Tensor:
    assert not torch.is_floating_point(obs)
    return obs.float() / 255 - 0.5



class ReplayBuffer:
    def __init__(self, ta_dim, max_size: int = 100000, seq_len: int = 2):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.data = TensorDict(
            {
                "obs": torch.zeros(max_size, 3, 3, 64, 64, dtype=torch.uint8),
                "ta": torch.zeros(max_size, dtype=torch.uint8),   #"ta": torch.zeros(max_size, ta_dim),
                #"done": torch.zeros(max_size, 1),
                #"rewards": torch.zeros(max_size, 1),
                #"ep_returns": torch.zeros(max_size, 1),
                #"values": torch.zeros(max_size, 1)
            },
            batch_size=max_size,
            device="cpu"
        )
        self.td_unfolded = _unfold_td(self.data, seq_len, 1)

    def add_single(self, obs, ta, done, rewards, ep_returns, values):
        obs = torch.from_numpy(obs).to(torch.uint8)
        ta = torch.from_numpy(ta).to(torch.uint8)
        self.data["obs"][self.ptr] = obs
        self.data["ta"][self.ptr] = ta
        #self.data["done"][self.ptr] = done
        #self.data["rewards"][self.ptr] = rewards
        #self.data["ep_returns"][self.ptr] = ep_returns
        #self.data["values"][self.ptr] = values

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add(self, obs, ta):
        """
        Add a batch of data to the replay buffer.
        """
        batch_size = obs.shape[0]
        if self.ptr + batch_size > self.max_size:
            # Split the batch if it crosses the end of the buffer
            first_part_size = self.max_size - self.ptr
            second_part_size = batch_size - first_part_size
            obs_first = obs[:first_part_size]
            obs_second = obs[first_part_size:]
            ta_first = ta[:first_part_size]
            ta_second = ta[first_part_size:]
            #done_first = done[:first_part_size]
            #done_second = done[first_part_size:]
            #rewards_first = rewards[:first_part_size]
            #rewards_second = rewards[first_part_size:]
            #ep_returns_first = ep_returns[:first_part_size]
            #ep_returns_second = ep_returns[first_part_size:]
            #values_first = values[:first_part_size]
            #values_second = values[first_part_size:]
            # Add the first part to the end of the buffer
            self.data["obs"][self.ptr:self.max_size] = torch.from_numpy(obs_first).to(torch.uint8)
            self.data["ta"][self.ptr:self.max_size] = torch.from_numpy(ta_first).to(torch.uint8)
            #self.data["done"][self.ptr:self.max_size] = done_first
            #self.data["rewards"][self.ptr:self.max_size] = rewards_first
            #self.data["ep_returns"][self.ptr:self.max_size] = ep_returns_first
            #self.data["values"][self.ptr:self.max_size] = values_first
            # Add the second part to the beginning of the buffer
            self.data["obs"][:second_part_size] = torch.from_numpy(obs_second).to(torch.uint8)
            self.data["ta"][:second_part_size] = torch.from_numpy(ta_second).to(torch.uint8)
            #self.data["done"][:second_part_size] = done_second
            #self.data["rewards"][:second_part_size] = rewards_second
            #self.data["ep_returns"][:second_part_size] = ep_returns_second
            #self.data["values"][:second_part_size] = values_second
            self.ptr = second_part_size
        else:
            # Add the entire batch at once
            self.data["obs"][self.ptr:self.ptr + batch_size] = torch.from_numpy(obs).to(torch.uint8)
            self.data["ta"][self.ptr:self.ptr + batch_size] = torch.from_numpy(ta).to(torch.uint8)
            #self.data["done"][self.ptr:self.ptr + batch_size] = done
            #self.data["rewards"][self.ptr:self.ptr + batch_size] = rewards
            #self.data["ep_returns"][self.ptr:self.ptr + batch_size] = ep_returns
            #self.data["values"][self.ptr:self.ptr + batch_size] = values
            self.ptr = (self.ptr + batch_size) % self.max_size
        self.size = min(self.size + batch_size, self.max_size)

    def get_dataloader(self, batch_size):
        valid_data = self.data[:self.size]
        dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
        return dataloader


    def get_iter_originalunfold(
        self,
        batch_size: int,
        device,
        shuffle=True,
        drop_last=True,
    ) -> Generator[TensorDict, None, None]:
        dataloader = DataLoader(
            self.td_unfolded[:self.size-1],
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=lambda x: x,
        )

        while True:
            for batch in dataloader:
                batch = batch.to(device)
                batch["obs"] = normalize_obs(batch["obs"])
                yield batch

    def get_iter(
        self,
        batch_size: int,
        device,
        shuffle=True,
        drop_last=True,
    ) -> Generator[TensorDict, None, None]:
        dataloader = DataLoader(
            self.data[:self.size],
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=lambda x: x,
        )

        while True:
            for batch in dataloader:
                batch = batch.to(device)
                batch["obs"] = normalize_obs(batch["obs"])
                yield batch