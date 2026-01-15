import torch
import torch.nn as nn

class SinkhornKnopp(nn.Module):
    """
    Sinkhorn-Knopp projection onto doubly stochastic matrices.

    Projects any matrix onto the Birkhoff polytope (set of doubly stochastic matrices) using alternating row and column normalization.

    Args:
        iterations: Number of normalization iterations (default: 20)
        eps: Small value for numerical stability (default: 1e-8)

    Returns:
        torch.Tensor: Output doubly stochastic tensor of shape (N, N).
    """
    def __init__(self, iterations: int = 20, eps: float = 1e-8):
        super(SinkhornKnopp, self).__init__()

        self.iterations = iterations
        self.eps = eps
    
    def forward(self, matrix: torch.Tensor) -> torch.Tensor:
        # Subtract max for numerical stability before exp
        P = torch.exp(matrix - matrix.max(dim=-1, keepdim=True).values)

        for _ in range(self.iterations):
            # Row Normalization
            P = P / (P.sum(axis=-1, keepdim=True) + self.eps)
            # Column Normalization
            P = P / (P.sum(axis=-2, keepdim=True) + self.eps)
        return P

class mHCResidual(nn.Module):
    """
    Manifold-Constrained Hyper-Connection residual module.

    Implements the mHC residual connection with learnable mixing matrices that are projected onto doubly stochastic matrices via Sinkhorn-Knopp.

    Args:
        dim: Hidden dimension size
        n_streams: Number of parallel streams (default: 4)
        sinkhorn_iters: Number of Sinkhorn iterations (default: 20)

    Returns:
        torch.Tensor: Output updated hidden state of shape (batch, seq_len, n_streams, dim).
    """
    def __init__(self, dim: int, n_streams: int = 4, sinkhorn_iters: int = 20):
        super(mHCResidual, self).__init__()

        self.dim = dim
        self.n_streams = n_streams

        self.sinkhorn = SinkhornKnopp(iterations=sinkhorn_iters)

        self.H_res = nn.Parameter(torch.randn(n_streams, n_streams) * 0.01)
        self.H_pre = nn.Parameter(torch.ones(1, n_streams) / n_streams)
        self.H_pos = nn.Parameter(torch.ones(n_streams, 1) / n_streams)

        self.alpha_res = nn.Parameter(torch.tensor(0.01))
        self.alpha_pre = nn.Parameter(torch.tensor(0.01))
        self.alpha_pos = nn.Parameter(torch.tensor(0.01))

        self.bias_res = nn.Parameter(torch.zeros(n_streams, dim))
        self.bias_pos = nn.Parameter(torch.zeros(n_streams, dim))
    
    def forward(self, x: torch.Tensor, layer_output: torch.Tensor) -> torch.Tensor:
        _, _, S, D = x.shape
        H_res_proj = self.sinkhorn(self.H_res)
        x_mixed = torch.einsum('btsd,sr->btrd', x, H_res_proj)
        x_mixed = self.alpha_res * x_mixed + self.bias_res.view(1, 1, S, D)

        layer_out = layer_output.unsqueeze(2) * self.H_pos.view(1, 1, S, 1)
        layer_out = self.alpha_pos * layer_out + self.bias_pos.view(1, 1, S, D)

        output = x + x_mixed + layer_out
        return output
    
    def get_aggregated_input(self, x: torch.Tensor) -> torch.Tensor:
        H_pre = torch.softmax(self.H_pre, dim=-1)
        aggregated = torch.einsum('btsd,os->btd', x, H_pre)
        return self.alpha_pre * aggregated
    
class mHCBlock(nn.Module):
    """
    Wrapper that adds mHC residual connections to any layer.

    Args:
        layer: The layer module to wrap
        dim: Hidden dimension
        n_streams: Number of parallel streams (default: 4)
        sinkhorn_iters: Number of Sinkhorn iterations (default: 20)

    Returns:
        torch.Tensor: Output tensor of shape (batch, n_streams, dim).
    """
    def __init__(self, layer: nn.Module, dim: int, n_streams: int = 4, sinkhorn_iters: int = 20):
        super(mHCBlock, self).__init__()

        self.layer = layer
        self.mhc = mHCResidual(dim, n_streams, sinkhorn_iters)
        self.n_streams = n_streams
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layer_input = self.mhc.get_aggregated_input(x)
        layer_output = self.layer(layer_input)
        output = self.mhc(x, layer_output)
        return output
