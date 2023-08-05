from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional
from types import FunctionType
import torch
import torch.nn as nn
from transformers import RobertaModel

class ConvStemConfig(NamedTuple):
    out_channels: int
    kernel_size: int
    stride: int
    norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d
    activation_layer: Callable[..., nn.Module] = nn.ReLU

def _log_api_usage_once(obj: Any) -> None:

    """
    Logs API usage(module and name) within an organization.
    In a large ecosystem, it's often useful to track the PyTorch and
    TorchVision APIs usage. This API provides the similar functionality to the
    logging module in the Python stdlib. It can be used for debugging purpose
    to log which methods are used and by default it is inactive, unless the user
    manually subscribes a logger via the `SetAPIUsageLogger method <https://github.com/pytorch/pytorch/blob/eb3b9fe719b21fae13c7a7cf3253f970290a573e/c10/util/Logging.cpp#L114>`_.
    Please note it is triggered only once for the same API call within a process.
    It does not collect any data from open-source users since it is no-op by default.
    For more information, please refer to
    * PyTorch note: https://pytorch.org/docs/stable/notes/large_scale_deployments.html#api-usage-logging;
    * Logging policy: https://github.com/pytorch/vision/issues/5052;
    Args:
        obj (class instance or method): an object to extract info from.
    """
    module = obj.__module__
    if not module.startswith("torchvision"):
        module = f"torchvision.internal.{module}"
    name = obj.__class__.__name__
    if isinstance(obj, FunctionType):
        name = obj.__name__
    torch._C._log_api_usage_once(f"{module}.{name}")

class MLP(torch.nn.Sequential):
    """This block implements the multi-layer perceptron (MLP) module.
    Args:
        in_channels (int): Number of channels of the input
        hidden_channels (List[int]): List of the hidden channel dimensions
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the linear layer. If ``None`` this layer won't be used. Default: ``None``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the linear layer. If ``None`` this layer won't be used. Default: ``torch.nn.ReLU``
        inplace (bool, optional): Parameter for the activation layer, which can optionally do the operation in-place.
            Default is ``None``, which uses the respective default values of the ``activation_layer`` and Dropout layer.
        bias (bool): Whether to use bias in the linear layer. Default ``True``
        dropout (float): The probability for the dropout layer. Default: 0.0
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        inplace: Optional[bool] = None,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        # The addition of `norm_layer` is inspired from the implementation of TorchMultimodal:
        # https://github.com/facebookresearch/multimodal/blob/5dec8a/torchmultimodal/modules/layers/mlp.py
        params = {} if inplace is None else {"inplace": inplace}

        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer(**params))
            layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        layers.append(torch.nn.Dropout(dropout, **params))

        super().__init__(*layers)
        _log_api_usage_once(self)

class MLPBlock(MLP):
    """Transformer MLP block."""

    _version = 2

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # Replacing legacy MLPBlock with MLP. See https://github.com/pytorch/vision/pull/6053
            for i in range(2):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}linear_{i+1}.{type}"
                    new_key = f"{prefix}{3*i}.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

class ConvEncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # depth-wise Separable conv
        self.depthwise_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim)
        self.pointwise_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, stride=1)
        #self.depthwise_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x = x.permute(0,2,1)
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = x.permute(0,2,1)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)

        x = x + input
        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y

class ConvEncoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = ConvEncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))

class MSLTransformer(nn.Module):
    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6)
    ):
        super().__init__()
        _log_api_usage_once(self)
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        self.conv_encoder = ConvEncoder(
            seq_length+1,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )

        classifier_heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if representation_size is None:
            classifier_heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
        else:
            classifier_heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
            classifier_heads_layers["act"] = nn.Tanh()
            classifier_heads_layers["head"] = nn.Linear(representation_size, num_classes)

        self.classifier_heads = nn.Sequential(classifier_heads_layers)

        #if hasattr(self.classifier_heads, "pre_logits") and isinstance(self.classifier_heads.pre_logits, nn.Linear):
        #    fan_in = self.classifier_heads.pre_logits.in_features
        #    nn.init.trunc_normal_(self.classifier_heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
        #    nn.init.zeros_(self.classifier_heads.pre_logits.bias)

        #if isinstance(self.classifier_heads.head, nn.Linear):
        #    nn.init.zeros_(self.classifier_heads.head.weight)
        #    nn.init.zeros_(self.classifier_heads.head.bias)

        # Snippet Regressor Linear Head
        self.snippet_heads = nn.Linear(hidden_dim, num_classes)
        #nn.init.zeros_(self.snippet_heads.weight)
        #nn.init.zeros_(self.snippet_heads.bias)
        self.sigmod = nn.Sigmoid()


    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        #x = self._process_input(x)
        n = x.shape[0]
        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        # CTEx2
        x = self.conv_encoder(x)

        # Classifier "token" as used by standard language architectures
        # Video Classifier Linear Head
        p = self.classifier_heads(x[:, 0]).squeeze()
        p = self.sigmod(p)

        # Snippet Regressor Linear Head
        x = self.snippet_heads(x[:, 1:]).squeeze()
        x = self.sigmod(x)

        return x,p

class MSLNet(nn.Module):
    def __init__(self, layer, winlen, k):
        super(MSLNet, self).__init__()
        self.k = k
        # embedding
        self.roberta_layer = RobertaModel.from_pretrained("roberta-base")
        # Instance Encoder
        self.conv1 = nn.Conv1d(768, 768, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(2, stride=2, ceil_mode=True)
        # Transformer-based Multi-SequenceLearning Network
        self.transformer_msl = MSLTransformer(
            seq_length=winlen-k+1,
            num_layers=layer,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=1024,
            dropout=0.4,
            attention_dropout=0.1,
            num_classes=1
        )

    def forward(self, wins_nodes_ids, wins_nodes_mask, device):
        # embedding
        x = torch.zeros(0).to(device)
        for nodes_ids, nodes_mask in zip(wins_nodes_ids, wins_nodes_mask):
            outputs = self.roberta_layer(nodes_ids, nodes_mask)
            xi = outputs["pooler_output"]

            xi_unfold = xi.unfold(0, self.k, 1)
            xi_unfold = self.conv1(xi_unfold)
            xi_unfold = self.pool1(xi_unfold)
            xi_unfold = xi_unfold.squeeze()

            x = torch.cat((x, xi_unfold.unsqueeze(0)))
        
        x,p = self.transformer_msl(x)

        return x,p
