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


def make_absvrp_instance(args):
    depot, loc, demand, capacity, *args = args
    assert not args

    partial = False

    locations = torch.tensor([depot] + loc, dtype=torch.float)
    demand = torch.tensor([-float("inf")] + demand, dtype=torch.float)

    kinds = torch.full(demand.shape, 1, dtype=int)
    kinds[0] = 0

    return {
        "partial": partial,
        "locations": locations,
        "demand": demand / capacity,
        "kinds": kinds,
    }


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

        if filename is not None:
            assert os.path.splitext(filename)[1] == ".pkl"

            data = pickle.load(open(filename, "rb"))
            self.data = list(
                map(
                    make_absvrp_instance,
                    data[offset : offset + num_samples],
                )
            )

            self.size = len(self.data)
            return

        # From VRP with RL paper https://arxiv.org/abs/1802.04240
        CAPACITIES = {10: 20.0, 20: 30.0, 50: 40.0, 100: 50.0}

        self.data = []

        n_depots = 1
        n_locations = size
        n_nodes = n_locations + n_depots

        def greedy_route(T):
            """Get a greedy route"""
            B = T.clone()

            path = [0]
            while len(path) < len(B):
                B[:, path[-1]] = float('inf')
                path.append(B[path[-1], :].argmin())
            path.append(0)

            return torch.tensor(path).long()

        for _ in range(num_samples):
            loc = torch.rand(n_nodes, 2)  # locations random in .uniform_(0, 1)
            pairs = torch.norm(loc.unsqueeze(0) - loc.unsqueeze(1), p=2, dim=-1)

            # node demands,  Uniform 1 - 9, scaled by capacities
            demand = torch.randint(1, 10, size=(n_nodes,)) / CAPACITIES[size]
            demand[0] = -float("inf")

            T = pairs
            D = demand
            Q = torch.ones(1) * CAPACITIES[size]

            router = greedy_route

            # sample time-windows
            vehicles = dict(enumerate(Q))
            nodes = dict(enumerate(D))

            clients = torch.arange(n_depots, n_nodes)
            depots = torch.arange(n_depots)

            n_circuits = len(vehicles)
            width = 1.0
            linger = 1.0

            # compute tentative visitation times
            times = torch.zeros(len(nodes))

            location = clients[torch.randperm(clients.shape[0])]
            starting = torch.randperm(depots.shape[0])[:n_circuits]

            for k, depot in enumerate(starting):
                # get the clients in the partition
                ix = torch.cat([torch.tensor([depot]), location[k::n_circuits]]).long()
                # get a plausible route (v_j)_{j=0}^n with v_n = v_0 through
                #  the clients in the group `ix`
                # XXX be careful not to ACCIDENTALLY transpose the matrix!
                route = ix[router(T[ix[:, None], ix[None, :]])]

                # compute the visitation times
                travel = T[route[:-1], route[1:]]  # XXX time from v_j to v_{j+1}, j=0..n-1
                times[route[1:]] = torch.flatten(travel + linger).cumsum(-1)  # XXX we overwrite depot times

            # compute cleaner depot times: the latest visitation time taking into
            #  account the return time
            index = clients[:, None]
            times[depots] = (T[index, depots[None, :]] + times[index]).max(0).values

            # generate symmetric windows around the arrival time, ignoring depots
            widths = torch.rand(len(times)) * width
            L, U = times - widths, times + widths
            L[depots] = 0.0

            tw = torch.stack([L, U], dim=-1)

            # node type tokens
            kinds = torch.empty(1 + size, dtype=int)
            kinds[0] = 0
            kinds[1:] = 1

            self.data.append(
                {
                    "partial": partial,
                    # "distances": pairs,
                    "locations": loc,
                    "demand": demand,
                    "kinds": kinds,
                    "tw": tw,
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
