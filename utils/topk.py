import torch


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


if __name__ == "__main__":
    groups, _ = torch.randint(0, 10, size=(200,)).sort()
    value = torch.randn(2, len(groups))

    idx = topk_indices(value, 4, dim=-1, largest=True, sorted=True, groups=groups)

    val = value[torch.arange(2).unsqueeze(1), idx]
    grp = groups[idx]
