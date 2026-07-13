"""Multi-task model core components."""
from .dsc import DepthwiseSeparableConv
from .encoder import DSCBottleneck, ResNet50DSCEncoder
from .soa import SOA
from .easpp import EASPP
from .decoder import DecoderBlock, LightweightDecoder
from .heads import SegmentationHead, SeverityHead, DepthHead, CameraHead
from .domain_adapter import GradientReversalLayer, DomainDiscriminator, DualDomainAdapter
from .multitask import MultiTaskModel

__all__ = [
    "DepthwiseSeparableConv",
    "DSCBottleneck",
    "ResNet50DSCEncoder",
    "SOA",
    "EASPP",
    "DecoderBlock",
    "LightweightDecoder",
    "SegmentationHead",
    "SeverityHead",
    "DepthHead",
    "CameraHead",
    "GradientReversalLayer",
    "DomainDiscriminator",
    "DualDomainAdapter",
    "MultiTaskModel",
]
