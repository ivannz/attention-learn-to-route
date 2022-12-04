# import math
import torch

from typing import NamedTuple
from torch import Tensor

_unused = object()


class AbsCVRP(NamedTuple):
    # B -- batch, S -- search (beamsearch)
    # [B] boolean flag, specifying if demands can be partially satisfied
    partial: Tensor
    # [B x N x N] large dense shortest path matrix used as reference
    distances: Tensor
    # [S] index to instance correspondence S -->> B surjective
    instance: Tensor
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
    tw: Tensor
    travel_time: Tensor

    done: Tensor
    # [S] the cumulative route cost
    cost: Tensor

    # class-level constant
    MAX_CAPACITY = 1.0
    MAX_TRAVEL_TIME = 1_000_000

    @classmethod
    def initialize(cls, input, visited_dtype=_unused):
        assert visited_dtype is _unused

        if "locations" in input:
            loc = input["locations"]
            dis = torch.norm(loc.unsqueeze(-3) - loc.unsqueeze(-2), p=2, dim=-1)

        elif "distances" in input:
            dis = input["distances"]

        else:
            raise TypeError(f"Bad CVRP problem instance `{list(input.keys())}`.")

        dem = input["demand"]
        assert dem.le(cls.MAX_CAPACITY).all()  # must not exceed unit capacity

        # tw = input["tw"]
        tw = torch.randint(
            0,
            cls.MAX_TRAVEL_TIME,
            size=dem.shape + (2,)
        ).sort(-1).values

        assert tw[..., 1].le(cls.MAX_TRAVEL_TIME).all()
        assert tw[..., 0].ge(0).all()

        # start at the depot
        at_depot = dem[:, 0]  # full autorefill at the depot
        assert at_depot.le(0).logical_and(at_depot.isinf()).all()

        return cls(
            partial=input["partial"],
            distances=dis,
            instance=torch.arange(len(dis), device=dis.device),
            demand=dem,
            mask=dem.le(0),
            capacity=dem.new_zeros(len(dis)).fill_(cls.MAX_CAPACITY),
            tw=tw,
            travel_time=tw.new_zeros(len(dis)),
            done=dis.new_zeros(len(dis), dtype=bool),
            loc=dis.new_zeros(len(dis), dtype=torch.long),
            cost=dis.new_zeros(len(dis)), # 
        )

    def update(self, selected):
        assert len(self.instance) == len(selected)

        # update costs with links from the current location and to endpoints
        # d_{b v_b \to w_b} for `loc` v_b and `selected` w_b

        cost = self.cost.add(self.distances[self.instance, self.loc, selected])
        travel_time = self.travel_time.add(self.distances[self.instance, self.loc, selected])

        # we deliver to the `n`-th client as much as we can when visiting them
        picked = selected.unsqueeze(-1)
        demand_at = self.demand.gather(-1, picked).squeeze(-1)

        # the change in capacity: +ve we refilled, -ve we delivered
        delta = demand_at.clamp(
            min=self.capacity - self.MAX_CAPACITY,
            max=self.capacity,
        ).neg_()

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

        tw_satisfied = travel_time.unsqueeze(-1).ge(self.tw[..., 0])
        tw_satisfied = tw_satisfied.logical_and(travel_time.unsqueeze(-1).le(self.tw[..., 1]))

        satisfied.logical_and_(tw_satisfied)

        # f_{bi} = `i` forbidden at `b` if `(d_{bi} \leq 0) OR (c_b \leq 0)`
        mask = satisfied.logical_or(capacity.le(0.0).unsqueeze(-1))

        # if partial deliveries are NOT allowed, then forbid unserviceables
        #  f_{bi} |= (d_{b i} > c_b) AND NOT p_b
        not_partial = self.partial[self.instance].logical_not().unsqueeze(-1)
        mask.logical_or_(demand.gt(capacity.unsqueeze(-1)).logical_and_(not_partial))

        # finally allow the depot if not currently in it, unless all clients
        #  have been satisfied
        #  f_{b0} = (\ell_b = 0) AND (\cup_{i \neq 0} (d_{bi} > 0))
        mask[:, 0] = demand.gt(0.0).any(-1).logical_and_(at_depot)

        # re-raise the finish flag
        #  d_b |= (\ell_b = 0) AND (\cap_{i \neq 0} (d_{bi} \leq 0))
        done = satisfied.all(-1).logical_and_(at_depot).logical_or_(self.done)

        # return the updated state
        return self._replace(
            demand=demand,
            mask=mask,
            loc=selected.clone(),
            capacity=capacity,
            travel_time=travel_time,
            done=done,
            cost=cost,
        )

    def __getitem__(self, index):
        return self._replace(
            instance=self.instance[index],
            demand=self.demand[index],
            mask=self.mask[index],
            loc=self.loc[index],
            capacity=self.capacity[index],
            done=self.done[index],
            cost=self.cost[index],
        )

    @property
    def visited(self):
        return self.demand.le(0.0)

    @property
    def demands_with_depot(self):
        return self.demand.clamp_min(-self.MAX_CAPACITY).unsqueeze(1)

    @property
    def ids(self):
        return self.instance

    @property
    def dist(self):
        return self.distances

    def get_final_cost(self):
        assert self.done.all()
        return self.cost

    def all_finished(self):
        return self.done.all()

    def get_finished(self):
        return self.done

    def get_current_node(self):
        return self.loc.unsqueeze(-1)

    def get_mask(self):
        return self.mask.unsqueeze(1)

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
    if not splits:
        return indptr.new_empty(0)

    # get the indices of top-k in-group values (recurrence)
    return torch.cat(
        [
            topk_indices(group, k, dim, largest, sorted).add_(base)
            for base, group in zip(indptr, splits)
        ],
        dim=-1,
    )


class Beam(NamedTuple):
    # `parent` indicates which beam the current beam came from
    parent: Tensor
    action: Tensor
    score: Tensor
    state: None

    @classmethod
    def initialize(cls, state):
        return cls(
            parent=torch.arange(len(state.cost), device=state.cost.device),
            action=None,
            score=state.cost.new_zeros(len(state.cost)),
            state=state,
        )

    def expand(self, parent, action, score):
        # populate new states from parents and update each with an action
        state = self.state[parent].update(action)
        return type(self)(parent, action, score, state)

    def select(self, k, *, largest=True):
        # select the top-k scoring beams within each bundle, formed be grouping
        # beams by instance id of the associated problem in `.state`.
        top_k = topk_indices(
            self.score,
            k,
            largest=largest,
            groups=self.state.instance,
        )
        return type(self)(
            self.parent[top_k],
            None if self.action is None else self.action[top_k],
            self.score[top_k],
            self.state[top_k],
        )


def beam_search(state, k, propose=None, *, commit=id, largest=True):
    beam = Beam.initialize(state)
    commit(beam)

    # propose expansion for each beam, materialize them and then filter
    while not beam.state.done.all():
        parent, action, score = propose(beam)
        if len(parent) < 1:
            break

        beam = beam.expand(parent, action, score).select(k, largest=largest)
        commit(beam)

    return beam


def propose(beam, *, k=3, kind):
    """A dummy beam expansion proposal."""
    assert kind in ("greedy", "dummy")

    if kind == "greedy":
        raw = beam.state.distances[beam.state.instance, beam.state.loc].neg()
        logits = raw.masked_fill_(beam.state.mask, float("-inf"))

    else:
        raw = torch.randn_like(beam.state.demand)
        raw.masked_fill_(beam.state.mask, float("-inf"))
        logits = raw.log_softmax(-1)

    log_p, top_k = logits.topk(k, dim=-1, largest=True)

    parent = torch.arange(len(logits)).repeat_interleave(k)
    scores = log_p.add_(beam.score.unsqueeze(-1)).flatten()
    action = top_k.flatten()

    is_finite = scores.isfinite()
    return parent[is_finite], action[is_finite], scores[is_finite]


if __name__ == "__main__":
    from functools import partial

    # distances
    n_batch_size, n_loc = 64, 50
    e = torch.rand(n_batch_size, 1 + n_loc, 1 + n_loc).log_().neg_()
    e.diagonal(dim1=-2, dim2=-1)[:] = 0
    for j in range(e.shape[1]):
        torch.minimum(e, e[..., :, [j]] + e[..., [j], :], out=e)

    # demands
    x = torch.rand(len(e), 1 + n_loc)

    # the depot has infinite supply
    x[:, 0] = -float("inf")

    p = torch.randint(0, 2, size=(n_batch_size,), dtype=bool)

    # e[:] = e[[0]]
    # x[:] = x[[0]]
    input = dict(distances=e, demand=x, partial=p)

    state = AbsCVRP.initialize(input)
    history = []
    out = beam_search(
        state,
        k=9,
        propose=partial(propose, k=9, kind="greedy"),
        commit=history.append,
    )

    # backtrack thru the saved beam history and rebuild the sequence
    #  $\pi_{t:} = \pi_t \circ \pi_{t+1:}$
    actions = []
    beam = last = out.select(1, largest=True)
    parent, score = beam.parent, beam.score
    # parent, score = slice(None), history[-1].score
    while history:
        beam = history.pop()
        if beam.action is None:
            break

        # get parents' actions and their parents
        actions.append(beam.action[parent])
        parent = beam.parent[parent]

    actions = torch.stack(actions[::-1], dim=-1)

    self = AbsCVRP.initialize(input)
    actions = []
    while not self.done.all():
        group, allow = self.mask.logical_not().nonzero(as_tuple=True)
        split = allow.split(group.unique_consecutive(return_counts=True)[1].tolist())
        selected = torch.cat([act[torch.randperm(len(act))][:1] for act in split])

        self, omega = self.update(selected), self
        actions.append(selected)

    actions = torch.stack(actions, dim=-1)
