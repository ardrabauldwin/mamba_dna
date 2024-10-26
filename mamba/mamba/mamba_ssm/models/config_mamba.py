from dataclasses import dataclass, field
import json

@dataclass
class MambaConfig:

    d_model: int = 2560
    d_intermediate: int = 0
    n_layer: int = 64
    vocab_size: int = 50277
    ssm_cfg: dict = field(default_factory=dict)
    attn_layer_idx: list = field(default_factory=list)
    attn_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    tie_embeddings: bool = True

    # added by me
    def to_json_string(self, use_diff: bool = True) -> str:
        """
        Serializes this instance to a JSON string.

        Args:
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `PretrainedConfig()`
                is serialized to JSON string.

        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """

        return json.dumps(
                self.to_dict()
                )
    
    # added by me
    def to_dict(self):
        """
        Transforms all attributes to a dictionary.
        
        """
        return {"d_model": self.d_model,
                "d_intermediate": self.d_intermediate,
                "n_layer": self.n_layer,
                "vocab_size": self.vocab_size,
                "ssm_cfg": self.ssm_cfg,
                "attn_layer_idx": self.attn_layer_idx,
                "attn_cfg": self.attn_cfg,
                "rms_norm": self.rms_norm,
                "residual_in_fp32": self.residual_in_fp32,
                "fused_add_norm": self.fused_add_norm,
                "pad_vocab_size_multiple": self.pad_vocab_size_multiple,
                "tie_embeddings": self.tie_embeddings}
