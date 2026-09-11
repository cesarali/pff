from typing import Optional

from pff.config_classes.node_pk_config import NodePKExperimentConfig
from pff.models.architectures.decoders_pk import (
    ODEDecoder,
    RNNDecoder,
    TransformerDecoder,
    TransformerDecoderZiQuery,
)
from pff.models.architectures.vector_fields_pk import TransformerVectorField


def get_decoder(model_config: NodePKExperimentConfig, *, init_dim: Optional[int] = None):
    """Instantiate the decoder specified in ``model_config``.

    Parameters
    ----------
    model_config:
        Complete NodePK configuration describing the decoder architecture.
    init_dim:
        Optional override for the conditioning feature dimensionality.  When
        omitted the decoder will default to conditioning on
        ``[init_state, first_t_s, dose, route]`` → 4 features.
    """

    decoder_name = None
    if getattr(model_config, "vector_field", None) is not None:
        decoder_name = "TransformerVectorField"
    else:
        decoder_name = model_config.network.decoder_name

    if decoder_name == "RNNDecoder":
        return RNNDecoder(model_config, init_dim=init_dim)
    if decoder_name == "ODEDecoder":
        return ODEDecoder(model_config, init_dim=init_dim)
    if decoder_name == "TransformerDecoder":
        return TransformerDecoder(model_config, init_dim=init_dim)
    if decoder_name == "TransformerDecoderZiQuery":
        return TransformerDecoderZiQuery(model_config, init_dim=init_dim)
    if decoder_name == "TransformerVectorField":
        return TransformerVectorField(model_config, init_dim=init_dim)
    else:
        raise Exception("Encoder Not Defined")
