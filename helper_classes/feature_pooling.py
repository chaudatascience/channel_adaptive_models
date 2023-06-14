from utils import ExtendedEnum


class FeaturePooling(str, ExtendedEnum):
    AVG = "avg"
    MAX = "max"
    AVG_MAX = "avg_max"
    NONE = "none"

