"""Encoder-decoder transformer producing multi-step quantile forecasts of streamflow."""
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowTransformer(nn.Module):
    """
    Encoder : history of HDE rainfall + groundwater (+ missing masks), patched into tokens
              (e.g. 24 x 15 min -> 6 h) so a 14-day window stays a few tens of tokens.
    Decoder : one token per forecast bin (6 h) over the next `decoder_days`, carrying GEFS rainfall mean/std,
              (optionally) GeoSmart groundwater mean/std and seasonality. Decoder tokens self-attend (forecasts are
              all known at the origin, so no causal mask is needed) and cross-attend to the encoder memory.
    Head    : the first `n_output_tokens` decoder tokens each emit `token_steps` flow steps x n_quantiles, giving
              horizon_steps = n_output_tokens * token_steps values per quantile (e.g. 56 tokens x 1 = 56 at 6-hourly output).
              Quantiles are made non-crossing by predicting the lowest quantile plus cumulative positive increments.
    """

    def __init__(
        self,
        enc_in: int,
        dec_in: int,
        history_steps: int,
        n_decoder_tokens: int,
        n_output_tokens: int,
        token_steps: int,
        quantiles: Sequence[float],
        d_model: int = 64,
        n_heads: int = 4,
        n_encoder_layers: int = 2,
        n_decoder_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        encoder_patch_steps: int = 4,
        n_catchments: int = 1,
        geosmart_skip_index: int | None = None,
    ):
        super().__init__()
        if history_steps % encoder_patch_steps:
            raise ValueError("history_steps must be divisible by encoder_patch_steps")
        if n_output_tokens > n_decoder_tokens:
            raise ValueError("n_output_tokens cannot exceed n_decoder_tokens")
        if list(quantiles) != sorted(quantiles):
            raise ValueError("quantiles must be sorted ascending")

        self.patch = encoder_patch_steps
        self.n_output_tokens = n_output_tokens
        self.token_steps = token_steps
        self.n_catchments = n_catchments
        self.n_q = len(quantiles)
        self.register_buffer("quantiles", torch.tensor(list(quantiles), dtype=torch.float32))

        n_enc_tokens = history_steps // encoder_patch_steps
        self.enc_proj = nn.Linear(enc_in * encoder_patch_steps, d_model)
        self.enc_pos = nn.Parameter(torch.randn(1, n_enc_tokens, d_model) * 0.02)
        self.dec_proj = nn.Linear(dec_in, d_model)
        self.dec_pos = nn.Parameter(torch.randn(1, n_decoder_tokens, d_model) * 0.02)
        self.catchment_embedding = nn.Embedding(n_catchments, d_model) if n_catchments > 1 else None

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, n_heads, dim_feedforward, dropout, batch_first=True, norm_first=True
            ),
            n_encoder_layers,
            enable_nested_tensor=False,
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model, n_heads, dim_feedforward, dropout, batch_first=True, norm_first=True
            ),
            n_decoder_layers,
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, token_steps * self.n_q)
        # Optional skip: the scaled GeoSmart value at each output token is mapped straight onto that token's raw outputs
        self.geosmart_skip_index = geosmart_skip_index
        self.geosmart_skip = None
        if geosmart_skip_index is not None:
            self.geosmart_skip = nn.Linear(1, token_steps * self.n_q, bias=False)
            nn.init.zeros_(self.geosmart_skip.weight)  # starts identical to the model without the skip

    def forward(
        self, enc: torch.Tensor, dec: torch.Tensor, catchment_id: torch.Tensor | None = None
    ) -> torch.Tensor:
        """enc (B, history_steps, enc_in), dec (B, n_decoder_tokens, dec_in) -> (B, horizon_steps, n_quantiles)."""
        b, t, f = enc.shape
        x = enc.reshape(b, t // self.patch, self.patch * f)
        enc_pos, dec_pos = self.enc_pos, self.dec_pos
        if self.catchment_embedding is not None:
            if catchment_id is None:
                raise ValueError("catchment_id is required when the model has catchment embeddings")
            embedding = self.catchment_embedding(catchment_id.long()).unsqueeze(1)
            enc_pos, dec_pos = enc_pos + embedding, dec_pos + embedding
        memory = self.encoder(self.enc_proj(x) + enc_pos)

        q = self.decoder(self.dec_proj(dec) + dec_pos, memory)
        q = self.norm(q[:, : self.n_output_tokens])
        out = self.head(q)
        if self.geosmart_skip is not None:
            out = out + self.geosmart_skip(dec[:, : self.n_output_tokens, self.geosmart_skip_index : self.geosmart_skip_index + 1])
        raw = out.reshape(b, self.n_output_tokens * self.token_steps, self.n_q)

        base = raw[..., :1]
        increments = F.softplus(raw[..., 1:])
        return torch.cat([base, base + torch.cumsum(increments, dim=-1)], dim=-1)


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    """Mean quantile (pinball) loss. pred (B, H, Q), target (B, H)."""
    err = target.unsqueeze(-1) - pred
    return torch.maximum(quantiles * err, (quantiles - 1) * err).mean()
