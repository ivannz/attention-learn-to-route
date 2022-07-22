import math
import torch
from torch import Tensor

from torch import nn
from einops import rearrange, repeat


class EdgeWeightedMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_attention_heads: int,
        head_size: int = None,
        bias: bool = True,
    ) -> None:
        if head_size is None:
            head_size, rem = divmod(embedding_dim, num_attention_heads)
            if rem > 0:
                raise ValueError(
                    f"{embedding_dim} is not a multiple" f" of {num_attention_heads}."
                )

        super().__init__()
        self.embedding_dim = embedding_dim
        self.head_size = head_size
        self.num_attention_heads = num_attention_heads

        # (Q)uery (K)ey (V)alue projections
        self.qkv = nn.Linear(
            embedding_dim,
            3 * num_attention_heads * head_size,
            bias=False,
        )

        self.ekv = nn.Sequential(
            nn.Linear(
                1,
                2 * num_attention_heads * head_size,
                bias=True,
            ),
            # we need this nonlinearity here, since we're encoding
            nn.LeakyReLU(),
        )

        # re-projection with noise
        self.prj = nn.Linear(
            num_attention_heads * head_size,
            embedding_dim,
            bias=bias,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # (qkv) Kaiming-like uniform init based on head fan-out
        stdv = 1.0 / math.sqrt(self.head_size)
        self.qkv.weight.data.uniform_(-stdv, stdv)

        # (qkv) Kaiming-like uniform init based on head fan-out
        stdv = 1.0 / math.sqrt(self.embedding_dim)
        self.prj.weight.data.uniform_(-stdv, stdv)
        if self.prj.bias is not None:
            self.prj.bias.data.zero_()

    def forward(
        self,
        x: Tensor,
        e: Tensor,  # weighted adjacency matrix (infinite weights -- masked)
        mask: Tensor = None,
    ) -> Tensor:
        # qkv is x is `B N C`, below S is `stack x 3`, and H -- # of heads
        # XXX in non-self attention the query sequence might have different
        #  size, in which case we would have to make a separate layer for Q,
        #  e.g. x is `B N C` and q is `B M C`
        que, key, val = rearrange(
            self.qkv(x),
            "B N (S H D) -> S B H N () D",
            S=3,
            H=self.num_attention_heads,
        )

        # compute edge messages for individual vertices
        ekm, evm = rearrange(
            self.ekv(e.unsqueeze(-1)),
            "B N M (S H D) -> S B H N M D",
            S=2,
            H=self.num_attention_heads,
        )

        # scaled attention
        #  $a_{j t s} = \frac{q_{j t}^\top k_{j t s}}{\sqrt{d}}$
        #     `.einsum("bhnjd, bhnmd -> bhnm", que, key + ekm)`
        #  $\alpha_{j t s} = \softmax(a_{j t s} + (- \infty) m_{j t s})_{s=1}^n$
        #  $y_{j t} = \sum_s \alpha_{j t s} v_{j t s}$
        #     `.einsum("bhn_m, bhnmd -> bhnd", attn, val + evm)`
        # XXX `attn @ val -> out` gives [B H M N] @ [B H N D] -> [B H M D]
        dots = torch.matmul(que, (key + ekm).transpose(-1, -2)).squeeze(-2)
        if mask is not None:
            dots = dots.masked_fill(
                repeat(mask.to(bool), "B M N -> B H M N", H=1),
                -math.inf,
            )

        attn = torch.softmax(dots.div(math.sqrt(self.head_size)), dim=-1)
        hat = attn.unsqueeze(-2).matmul(val + evm).squeeze(-2)

        # dimshuffle and reproject
        return self.prj(rearrange(hat, "B H M D -> B M (H D)"))


class EdgeWeightedTransformerLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_attention_heads: int,
        intermediate_size: int,
        head_size: int = None,
        *,
        layernorm: nn.Module = nn.LayerNorm,
        gelu: nn.Module = nn.GELU,
        elementwise_affine: bool = True,
        postnormalize: bool = False,
    ) -> None:
        super().__init__()
        self.postnormalize = postnormalize

        # we use pre-norm MH self-A, but optionally disable the learnable affine
        # transformation in the normalizer before the MHA, since it makes sense
        # to think about the compatibilities in the attention matrix (pre-softmax)
        # as _semantic_ covariances. Hence we would want the normalizer NOT to
        # translate the input hiddens to an arbitrary location (scaling if ok,
        # since it actually means a learnable temperature).
        self.pn1 = layernorm(
            embedding_dim,
            elementwise_affine=elementwise_affine,
        )
        self.mha = EdgeWeightedMultiHeadSelfAttention(
            embedding_dim, num_attention_heads, head_size
        )
        self.pn2 = layernorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, intermediate_size),
            gelu(),
            nn.Linear(intermediate_size, embedding_dim),
        )

    def forward(self, x: Tensor, e: Tensor, mask: Tensor = None) -> Tensor:
        if self.postnormalize:
            x = self.pn1(x + self.mha(x, e, mask))
            x = self.pn2(x + self.mlp(x))
            return x

        x = x + self.mha(self.pn1(x), e, mask)
        x = x + self.mlp(self.pn2(x))
        return x


class BatchSeqNorm(nn.BatchNorm1d):
    def __init__(
        self,
        normalized_shape,
        eps=1e-05,
        elementwise_affine=True,
        device=None,
        dtype=None,
    ):
        super().__init__(
            normalized_shape,
            eps,
            momentum=0.1,
            affine=elementwise_affine,
            track_running_stats=True,
            device=device,
            dtype=dtype,
        )

    def forward(self, input):
        x = super().forward(rearrange(input, "B N D -> (B N) D"))
        return rearrange(x, "(B N) D -> B N D", B=len(input))


class EdgeWeightedGraphEncoder(nn.Module):
    def __init__(
        self,
        n_heads: int,
        embed_dim: int,
        n_layers: int,
        node_dim: int = None,
        normalization: str = None,
        feed_forward_hidden: int = 512,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            (
                EdgeWeightedTransformerLayer(
                    embed_dim,
                    n_heads,
                    feed_forward_hidden,
                    layernorm=BatchSeqNorm,
                    gelu=nn.ReLU,
                    elementwise_affine=True,
                    postnormalize=True,
                )
                for _ in range(n_layers)
            )
        )

    def forward(
        self,
        input: dict[str, Tensor],
        mask: Tensor = None,
    ) -> tuple[Tensor, Tensor]:

        h = input["nodes"]
        for layer in self.layers:
            h = layer(h, input["edges"])
        return (h, h.mean(dim=1))


if __name__ == "__main__":
    """Sanity check"""
    # prep some data
    e = torch.rand(3, 51, 51).log_().neg_()
    e.diagonal(dim1=-2, dim2=-1)[:] = 0
    for k in range(e.shape[1]):
        torch.minimum(e, e[..., :, [k]] + e[..., [k], :], out=e)

    x = torch.randn(3, 51, 128)

    self = EdgeWeightedGraphEncoder(8, 128, 3)
    out = self(
        {"nodes": x, "edges": e},
        torch.rand(3, 51, 51).lt(0.25),
    )

    print(out)
