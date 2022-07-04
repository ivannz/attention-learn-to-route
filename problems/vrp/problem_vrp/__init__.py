from torch.utils.data import Dataset
import torch
import os
import pickle

from problems.vrp.problem_vrp.cvrp import CVRP as BaseCVRP
from problems.vrp.problem_vrp.sdvrp import SDVRP as BaseSDVRP
from problems.vrp.problem_vrp.cvrp import AbsCVRP as BaseAbsCVRP


def make_instance(args):
    depot, loc, demand, capacity, *args = args
    grid_size = 1
    if len(args) > 0:
        depot_types, customer_types, grid_size = args
    return {
        "loc": torch.tensor(loc, dtype=torch.float) / grid_size,
        "demand": torch.tensor(demand, dtype=torch.float) / capacity,
        "depot": torch.tensor(depot, dtype=torch.float) / grid_size,
    }


class VRPDataset(Dataset):
    def __init__(
        self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None
    ):
        super(VRPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == ".pkl"

            with open(filename, "rb") as f:
                data = pickle.load(f)
            self.data = [
                make_instance(args) for args in data[offset : offset + num_samples]
            ]

        else:

            # From VRP with RL paper https://arxiv.org/abs/1802.04240
            CAPACITIES = {10: 20.0, 20: 30.0, 50: 40.0, 100: 50.0}

            self.data = [
                {
                    "loc": torch.FloatTensor(size, 2).uniform_(0, 1),
                    # Uniform 1 - 9, scaled by capacities
                    "demand": (torch.FloatTensor(size).uniform_(0, 9).int() + 1).float()
                    / CAPACITIES[size],
                    "depot": torch.FloatTensor(2).uniform_(0, 1),
                }
                for i in range(num_samples)
            ]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


class CVRP(BaseCVRP):
    @staticmethod
    def make_dataset(*args, **kwargs):
        return VRPDataset(*args, **kwargs)


class SDVRP(BaseSDVRP):
    @staticmethod
    def make_dataset(*args, **kwargs):
        return VRPDataset(*args, **kwargs)


class AbsVRPDataset(Dataset):
    def __init__(
        self,
        filename=None,
        size=50,
        partial=False,
        num_samples=1000000,
        offset=0,
        distribution=None,
    ):
        super().__init__()
        assert filename is None

        self.data_set = []
        # From VRP with RL paper https://arxiv.org/abs/1802.04240
        CAPACITIES = {10: 20.0, 20: 30.0, 50: 40.0, 100: 50.0}

        self.data = []
        for _ in range(num_samples):
            loc = torch.rand(1 + size, 2)  # locations random in .uniform_(0, 1)
            pairs = torch.norm(loc.unsqueeze(0) - loc.unsqueeze(1), p=2, dim=-1)

            # node demands,  Uniform 1 - 9, scaled by capacities
            demand = torch.randint(1, 10, size=(1 + size,)) / CAPACITIES[size]
            demand[0] = -float("inf")

            # node type toekns
            kinds = torch.empty(1 + size, dtype=int)
            kinds[0] = 0
            kinds[1:] = 1

            self.data.append(
                {
                    "partial": partial,
                    "distances": pairs,
                    "demand": demand,
                    "kinds": kinds,
                }
            )

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


class AbsCVRP(BaseAbsCVRP):
    NAME = "abscvrp"  # Abstract Capacitated Vehicle Routing Problem

    @staticmethod
    def make_dataset(*args, **kwargs):
        return AbsVRPDataset(*args, **kwargs)
