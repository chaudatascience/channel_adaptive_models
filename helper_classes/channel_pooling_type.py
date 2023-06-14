from utils import ExtendedEnum


class ChannelPoolingType(str, ExtendedEnum):
    AVG = "avg"
    SUM = "sum"
    WEIGHTED_SUM_RANDOM = "weighted_sum_random"
    WEIGHTED_SUM_ONE = "weighted_sum_one"
    WEIGHTED_SUM_RANDOM_NO_SOFTMAX = "weighted_sum_random_no_softmax"
    WEIGHTED_SUM_ONE_NO_SOFTMAX = "weighted_sum_one_no_softmax"
    WEIGHTED_SUM_RANDOM_PAIRWISE_NO_SOFTMAX = "weighted_sum_random_pairwise_no_softmax"
    WEIGHTED_SUM_RANDOM_PAIRWISE = "weighted_sum_random_pairwise"

    ATTENTION = "attention"



