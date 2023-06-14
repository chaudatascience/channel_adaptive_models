from utils import ExtendedEnum


class FirstLayerInit(str, ExtendedEnum):
    ## make attribute capital
    REINIT_AS_RANDOM = "reinit_as_random"
    PRETRAINED_PAD_AVG = "pretrained_pad_avg"  # pad with avg of pretrained weights for additional channels
    PRETRAINED_PAD_RANDOM = (
        "pretrained_pad_random"  # pad with random values for additional channels
    )
    PRETRAINED_PAD_DUPS = (
        "pretrained_pad_dups"  # pad with duplicates channels for additional channels
    )
