import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ExpertMLP(nn.Module):
    """Single expert FFN (SwiGLU-style: gate + up -> silu -> down)."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoELayer(nn.Module):
    """
    Mixture of Experts layer (Mixtral-compatible).

    Uses a router (gate) to select top-k experts per token.
    Tensor names follow Mixtral conventions for GGUF compatibility:
    - gate -> router linear
    - experts[i].gate_proj, up_proj, down_proj -> expert FFNs
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_local_experts: int = 8,
        num_experts_per_tok: int = 2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok

        self.gate = nn.Linear(hidden_size, num_local_experts, bias=False)
        self.experts = nn.ModuleList(
            [ExpertMLP(hidden_size, intermediate_size) for _ in range(num_local_experts)]
        )

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        router_logits = self.gate(hidden_states_flat)
        routing_weights = F.softmax(router_logits, dim=-1)

        topk_weights, topk_indices = torch.topk(
            routing_weights, self.num_experts_per_tok, dim=-1
        )
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        final_hidden_states = torch.zeros_like(hidden_states_flat)

        expert_mask = F.one_hot(topk_indices, num_classes=self.num_local_experts)
        expert_mask = expert_mask.permute(2, 1, 0)  # [num_experts, top_k, num_tokens]

        for expert_idx in range(self.num_local_experts):
            expert = self.experts[expert_idx]
            mask = expert_mask[expert_idx]  # [top_k, num_tokens]

            for top_k_idx in range(self.num_experts_per_tok):
                token_indices = mask[top_k_idx].nonzero(as_tuple=True)[0]
                if token_indices.numel() == 0:
                    continue

                expert_input = hidden_states_flat[token_indices]
                expert_output = expert(expert_input)
                weights = topk_weights[token_indices, top_k_idx].unsqueeze(-1)
                final_hidden_states[token_indices] += expert_output * weights

        final_hidden_states = final_hidden_states.view(batch_size, seq_len, hidden_dim)
        router_aux_loss = self._load_balancing_loss(router_logits, topk_indices)

        return final_hidden_states, router_aux_loss

    def _load_balancing_loss(
        self, router_logits: torch.Tensor, topk_indices: torch.Tensor
    ) -> torch.Tensor:
        """Compute load balancing auxiliary loss to encourage even expert usage."""
        num_tokens = router_logits.shape[0]
        routing_probs = F.softmax(router_logits, dim=-1)

        expert_mask = F.one_hot(topk_indices, num_classes=self.num_local_experts)
        expert_usage = expert_mask.float().sum(dim=1).mean(dim=0)
        expert_probs = routing_probs.mean(dim=0)

        aux_loss = (expert_usage * expert_probs).sum() * self.num_local_experts
        return aux_loss
