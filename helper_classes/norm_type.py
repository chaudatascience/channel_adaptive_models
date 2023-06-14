from utils import ExtendedEnum


class NormType(str, ExtendedEnum):
    BATCH_NORM = "batch_norm"
    LAYER_NORM = "layer_norm"
    INSTANCE_NORM = "instance_norm"
    BATCH_NORM_INVERSE = "batch_norm_inverse"


if __name__ == '__main__':
    print("NormType.BATCH_NORM", NormType.BATCH_NORM)
    print(NormType.BATCH_NORM == "batch_norm")
