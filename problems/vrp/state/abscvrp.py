import torch

from typing import NamedTuple
from torch import Tensor

_unused = object()


class AbsCVRP(NamedTuple):
    # B -- batch, S -- search (beamsearch)
    # boolean flag, specifying if demands can be partially satisfied
    partial: bool
    # [B x N x N] large dense shortest path matrix used as reference
    distances: Tensor
    # [S] the instance at each beam S -->> B
    beam: Tensor
    # [S x ...] the runtime state
    # [S x N] residual demand
    demand: Tensor
    # [S x N] mask of forbidden actions
    mask: Tensor
    # [S] current location of the vehicle
    loc: Tensor
    # [S] current capacity (single vehicle)
    capacity: Tensor
    # [S] flag indicating if the simulation has terminated
    done: Tensor
    # [S] the cumulative costs
    costs: Tensor

    # class-level constant
    MAX_CAPACITY = 1.0

    @classmethod
    def initialize(cls, input, visited_dtype=_unused):
        assert visited_dtype is _unused

        dis, dem = input["distances"], input["demand"]
        assert dem.le(cls.MAX_CAPACITY).all()  # must not exceed unit capacity

        # start at the depot
        at_depot = dem[:, 0]  # full autorefill at the depot
        assert at_depot.le(0).logical_and(at_depot.isinf()).all()

        return cls(
            partial=input["partial"],
            distances=dis,
            beam=torch.arange(len(dis), device=dis.device),
            demand=dem,
            mask=dem.le(0),
            capacity=dis.new_zeros(len(dis)).fill_(cls.MAX_CAPACITY),
            done=dis.new_zeros(len(dis), dtype=bool),
            loc=dis.new_zeros(len(dis), dtype=torch.long),
            costs=dis.new_zeros(len(dis)),
        )

    def update(self, selected):
        # update costs with links from the current location and to endpoints
        # d_{b v_b \to w_b} for `loc` v_b and `selected` w_b
        costs = self.costs.add(self.distances[self.beam, self.loc, selected])

        # we deliver to the `n`-th client as much as we can when visiting them
        picked = selected.unsqueeze(-1)
        demand_at = self.demand.gather(-1, picked).squeeze(-1)

        # the change in capacity: +ve we refilled, -ve we delivered
        delta = demand_at.clamp(
            min=self.capacity - self.MAX_CAPACITY, max=self.capacity
        )
        delta.neg_()

        # compute new capacity and demands (produce copies)
        capacity = self.capacity.add(delta)
        demand = self.demand.scatter_add(-1, picked, delta.unsqueeze(-1))

        # compute the updated mask. masking rules:
        # 1) returning to the depot is always an option, unless we return
        #    to it twice in a row, when there is unsatisfied demand
        # 2) nodes with non-+ve residual demand are not to be revisited
        # 3) clients not serviceable with the current capacity cannot be visited
        # 4) FORCE return to depot if empty
        satisfied = demand.le(0.0)
        at_depot = selected.eq(0)

        mask = satisfied.logical_or(capacity.unsqueeze(-1).le(0.0))
        if self.partial:
            mask.logical_or_(demand.gt(capacity.unsqueeze(-1)))

        mask[:, 0] = demand.gt(0).any(-1).logical_and_(at_depot)
        done = satisfied.all(-1).logical_and_(at_depot).logical_or_(self.done)

        # return the updated state
        return self._replace(
            demand=demand,
            mask=mask,
            loc=selected.clone(),
            capacity=capacity,
            done=done,
            costs=costs,
        )

    def __getitem__(self, index):
        assert isinstance(index, (Tensor, slice))
        return self._replace(
            beam=self.beam[index],
            demand=self.demand[index],
            mask=self.mask[index],
            loc=self.loc[index],
            capacity=self.capacity[index],
            done=self.done[index],
            costs=self.costs[index],
        )

    @property
    def visited(self):
        return self.demand.le(0.0)

    @property
    def dist(self):
        return self.distances

    def get_final_cost(self):
        assert self.done.all()
        return self.costs

    def all_finished(self):
        return self.done.all()

    def get_finished(self):
        return self.done

    def get_current_node(self):
        return self.loc

    def get_mask(self):
        return self.mask

    def construct_solutions(self, actions):
        return actions


def topk_indices(input, k, dim=-1, largest=True, sorted=True, *, groups=None):
    if groups is None:
        return input.topk(
            min(k, input.shape[dim]),
            dim=dim,
            largest=largest,
            sorted=sorted,
        ).indices

    # detect edges, assuming groups are continuous
    edges = groups.new_ones(1 + len(groups), dtype=bool)
    edges[1:-1] = groups[1:].ne(groups[:-1])

    # get the start index of each group and do the splitting
    indptr = edges.nonzero()[:, 0]
    splits = input.split(indptr.diff().tolist(), dim)

    # get the indices of top-k in-group values (recurrence)
    return torch.cat(
        [
            topk_indices(group, k, dim, largest, sorted).add_(base)
            for base, group in zip(indptr, splits)
        ],
        dim=-1,
    )


class Beam(NamedTuple):
    parent: Tensor
    state: None
    score: Tensor
    action: Tensor

    @classmethod
    def initialize(cls, state):
        return cls(
            parent=state.beam.clone(),
            state=state,
            score=state.costs.new_zeros(state.beam.shape),
            action=None,
        )

    def expand(self, parent, action, score=None):
        return self._replace(
            parent=parent,
            state=self.state[parent].update(action),
            score=score,
            action=action,
        )

    def select(self, k, largest=True):
        idx = topk_indices(self.score, k, largest=largest, groups=self.state.beam)
        return self._replace(
            parent=None if self.parent is None else self.parent[idx],
            state=self.state[idx],
            score=self.score[idx],
            action=None if self.action is None else self.action[idx],
        )


def beam_search(state, n_beam_k, largest=True, propose=None, keep_states=False):
    beam = Beam.initialize(state)

    history = []
    append = history.append if keep_states else id

    append(beam)
    while not beam.state.done.all():
        parent, action, score = propose(beam)
        if parent is None:
            return history, None

        # Expand and update the state according to the selected actions
        beam = beam.expand(parent, action, score=score).select(n_beam_k, largest)
        append(beam)

    # Return the final state separately since beams may not keep state
    return history, beam.state


def propose(beam, k=3):
    raw = torch.randn_like(beam.state.demand).masked_fill_(
        beam.state.mask, float("-inf")
    )
    logits = raw.log_softmax(-1)

    log_p, top_k = logits.topk(k, dim=-1)

    # get expanded beam scores
    parent = torch.arange(len(top_k)).repeat_interleave(k)
    action = top_k.flatten()
    scores = torch.flatten(beam.score.unsqueeze(-1) + log_p)

    is_finite = scores.isfinite()

    return parent[is_finite], action[is_finite], scores[is_finite]


if __name__ == "__main__":
    # distances
    n_batch_size, n_loc = 64, 50
    e = torch.rand(n_batch_size, 1 + n_loc, 1 + n_loc).log_().neg_()
    e.diagonal(dim1=-2, dim2=-1)[:] = 0
    for k in range(e.shape[1]):
        torch.minimum(e, e[..., :, [k]] + e[..., [k], :], out=e)

    # demands
    x = torch.rand(len(e), 1 + n_loc)

    # the depot has infinite supply
    x[:, 0] = -float("inf")

    input = dict(distances=e, demand=x, partial=False)

    state = AbsCVRP.initialize(input)
    history, out = beam_search(state, 5, propose=propose, keep_states=True)

    self = AbsCVRP.initialize(input)
    actions = []
    while not self.done.all():
        group, allow = self.mask.logical_not().nonzero(as_tuple=True)
        split = allow.split(group.unique_consecutive(return_counts=True)[1].tolist())
        selected = torch.cat([act[torch.randperm(len(act))][:1] for act in split])

        self, omega = self.update(selected), self
        actions.append(selected)

    actions = torch.stack(actions, dim=-1)
