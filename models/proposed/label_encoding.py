CLASS_TO_INDEX = {
    "benign": 0,
    "attack": 1,
}

INDEX_TO_CLASS = {
    index: label_class
    for label_class, index in CLASS_TO_INDEX.items()
}


def encode_class_label(label):
    label_class = label.get("class")

    try:
        return CLASS_TO_INDEX[label_class]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported label class: {label_class!r}"
        ) from exc


def decode_class_index(index):
    try:
        return INDEX_TO_CLASS[int(index)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Unsupported class index: {index!r}"
        ) from exc
