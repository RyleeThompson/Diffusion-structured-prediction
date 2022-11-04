import torch as th


def int2softmax(classes, num_classes):
    return th.eye(num_classes, device=classes.device)[classes]


def softmax2int(classes):
    return classes.argmax(-1)


def int2bits(classes, num_classes=None, num_bits=None):
    assert num_bits is not None or num_classes is not None
    if num_bits is None:
        num_bits = num_classes2num_bits(num_classes)

    mask = 2 ** th.arange(num_bits).to(classes.device, classes.dtype)
    return classes.unsqueeze(-1).bitwise_and(mask).ne(0).byte().type(th.float32)


def int2scaledbits(classes, num_bits=None, num_classes=None):
    bits = int2bits(classes, num_bits=num_bits, num_classes=num_classes)
    scaled_bits = bits * 2 - 1
    return scaled_bits


def scaledbits2bits(scaled_bits):
    return (scaled_bits > 0).type(th.long)


def scaledbits2int(scaled_bits):
    bits = scaledbits2bits(scaled_bits)
    return bits2int(bits)


def bits2int(bits):
    powers = 2 ** th.arange(bits.shape[-1]).to(bits.device)
    return (bits * powers).sum(dim=-1)


def num_classes2num_bits(num_classes):
    return th.ceil(th.log2(th.tensor(num_classes))).item()
