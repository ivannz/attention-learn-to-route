import torch

from problems.vrp.state.cvrp import StateCVRP
from problems.vrp.state.abscvrp import AbsCVRP as StateAbsCVRP
from utils.beam_search import beam_search


class CVRP(object):

    NAME = "cvrp"  # Capacitated Vehicle Routing Problem

    VEHICLE_CAPACITY = 1.0  # (w.l.o.g. vehicle capacity is 1, demands should be scaled)

    @staticmethod
    def get_costs(dataset, pi):
        batch_size, graph_size = dataset["demand"].size()
        # Check that tours are valid, i.e. contain 0 to n -1
        sorted_pi = pi.data.sort(1)[0]

        # Sorting it should give all zeros at front and then 1...n
        assert (
            torch.arange(1, graph_size + 1, out=pi.data.new())
            .view(1, -1)
            .expand(batch_size, graph_size)
            == sorted_pi[:, -graph_size:]
        ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"

        # Visiting depot resets capacity so we add demand = -capacity (we make sure
        #  it does not become negative)
        demand_with_depot = torch.cat(
            (
                torch.full_like(dataset["demand"][:, :1], -CVRP.VEHICLE_CAPACITY),
                dataset["demand"],
            ),
            1,
        )
        d = demand_with_depot.gather(1, pi)

        used_cap = torch.zeros_like(dataset["demand"][:, 0])
        for i in range(pi.size(1)):
            used_cap += d[
                :, i
            ]  # This will reset/make capacity negative if i == 0, e.g. depot visited
            # Cannot use less than 0
            used_cap[used_cap < 0] = 0
            assert (
                used_cap <= CVRP.VEHICLE_CAPACITY + 1e-5
            ).all(), "Used more than capacity"

        # Gather dataset in order of tour
        loc_with_depot = torch.cat((dataset["depot"][:, None, :], dataset["loc"]), 1)
        d = loc_with_depot.gather(
            1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1))
        )

        # Length is distance (L2-norm of difference) of each next location to its
        #  prev and of first and last to depot
        return (
            (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
            + (d[:, 0] - dataset["depot"]).norm(p=2, dim=1)  # Depot to first
            + (d[:, -1] - dataset["depot"]).norm(
                p=2, dim=1
            )  # Last to depot, will be 0 if depot is last
        ), None

    @staticmethod
    def make_state(*args, **kwargs):
        return StateCVRP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(
        input,
        beam_size,
        expand_size=None,
        compress_mask=False,
        model=None,
        max_calc_batch_size=4096,
    ):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam,
                fixed,
                expand_size,
                normalize=True,
                max_calc_batch_size=max_calc_batch_size,
            )

        state = CVRP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


class AbsCVRP:
    NAME = "abscvrp"  # Abstract Capacitated Vehicle Routing Problem

    VEHICLE_CAPACITY = 1.0  # (w.l.o.g. vehicle capacity is 1, demands should be scaled)

    @classmethod
    def get_costs(cls, dataset, pi):
        batch_size, graph_size = dataset["demand"].size()
        # Check that tours are valid, i.e. contain 0 to n -1
        sorted_pi = pi.data.sort(1)[0]

        # Sorting it should give all zeros at front and then 1...n
        arange = torch.arange(0, graph_size, out=pi.data.new())
        assert (arange.unsqueeze(0) == sorted_pi[:, -graph_size:]).all() and (
            sorted_pi[:, :-graph_size] == 0
        ).all(), "Invalid tour"

        # clamp supply at -capacity
        dem = dataset["demand"].gather(1, pi)
        dem[dem.isinf()] = -cls.VEHICLE_CAPACITY

        used_cap = torch.zeros_like(dataset["demand"][:, 0])
        for i in range(pi.size(1)):
            used_cap += dem[:, i]
            # This will reset/make capacity negative if i == 0, e.g. depot visited
            # Cannot use less than 0
            used_cap[used_cap < 0] = 0
            assert (
                used_cap <= cls.VEHICLE_CAPACITY + 1e-5
            ).all(), "Used more than capacity"

        # gather dataset in order of tour, and pad with depots
        end = torch.full_like(pi[:, :1], 0)
        pad = torch.cat((end, pi, end), dim=-1)

        if "locations" in dataset:
            loc = dataset["locations"]
            dist = torch.norm(loc.unsqueeze(0) - loc.unsqueeze(1), p=2, dim=-1)

        elif "distances" in dataset:
            dist = dataset["distances"]

        else:
            raise TypeError(f"Bad CVRP problem instance `{list(input.keys())}`.")

        idx = torch.arange(len(dist), device=dist.device).unsqueeze(-1)
        return dist[idx, pad[:, :-1], pad[:, 1:]].sum(-1), None

    @staticmethod
    def make_state(*args, **kwargs):
        return StateAbsCVRP.initialize(*args, **kwargs)

    @classmethod
    def beam_search(
        cls,
        input,
        beam_size,
        expand_size=None,
        compress_mask=False,
        model=None,
        max_calc_batch_size=4096,
    ):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam,
                fixed,
                expand_size,
                normalize=True,
                max_calc_batch_size=max_calc_batch_size,
            )

        state = cls.make_state(input)
        return beam_search(state, beam_size, propose_expansions)
