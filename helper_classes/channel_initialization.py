from utils import ExtendedEnum


class ChannelInitialization(str, ExtendedEnum):
    ZERO = "zero"
    RANDOM = "random"
