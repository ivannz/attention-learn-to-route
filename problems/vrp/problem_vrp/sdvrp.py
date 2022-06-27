import torch

from problems.vrp.state.sdvrp import StateSDVRP
from utils.beam_search import beam_search


class SDVRP(object):

    NAME = "sdvrp"  # Split Delivery Vehicle Routing Problem

    VEHICLE_CAPACITY = 1.0  # (w.l.o.g. vehicle capacity is 1, demands should be scaled)

    @staticmethod
    def get_costs(dataset, pi):
        batch_size, graph_size = dataset["demand"].size()

        # Each node can be visited multiple times, but we always deliver as much
        #  demand as possible. We check that at the end all demand has been satisfied
        demands = torch.cat(
            (
                torch.full_like(dataset["demand"][:, :1], -SDVRP.VEHICLE_CAPACITY),
                dataset["demand"],
            ),
            1,
        )
        rng = torch.arange(batch_size, out=demands.data.new().long())
        used_cap = torch.zeros_like(dataset["demand"][:, 0])
        a_prev = None
        for a in pi.transpose(0, 1):
            assert (
                a_prev is None or (demands[((a_prev == 0) & (a == 0)), :] == 0).all()
            ), "Cannot visit depot twice if any nonzero demand"
            d = torch.min(demands[rng, a], SDVRP.VEHICLE_CAPACITY - used_cap)
            demands[rng, a] -= d
            used_cap += d
            used_cap[a == 0] = 0
            a_prev = a
        assert (demands == 0).all(), "All demand must be satisfied"

        # Gather dataset in order of tour
        loc_with_depot = torch.cat((dataset["depot"][:, None, :], dataset["loc"]), 1)
        d = loc_with_depot.gather(
            1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1))
        )

        # Length is distance (L2-norm of difference) of each next location to its prev
        #  and of first and last to depot
        return (
            (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
            + (d[:, 0] - dataset["depot"]).norm(p=2, dim=1)  # Depot to first
            + (d[:, -1] - dataset["depot"]).norm(
                p=2, dim=1
            )  # Last to depot, will be 0 if depot is last
        ), None

    @staticmethod
    def make_state(*args, **kwargs):
        return StateSDVRP.initialize(*args, **kwargs)

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
        assert not compress_mask, "SDVRP does not support compression of the mask"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam,
                fixed,
                expand_size,
                normalize=True,
                max_calc_batch_size=max_calc_batch_size,
            )

        state = SDVRP.make_state(input)

        return beam_search(state, beam_size, propose_expansions)
