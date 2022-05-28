import sys

import torch


def color_mapping(predictions):
    mapping_rules = {
                        0: 7,
                        1: 4,
                        2: 4,
                        3: 6,
                        4: 3,
                        5: 5,
                        6: 1,
                        7: 2,
                        8: 1,
                        9: 1,
                        10: 1,
                        11: 0
                    }
    max_val = 7
    out_predictions = torch.zeros([predictions.size(0), max_val + 1])
    for k, prediction in enumerate(predictions):
        out = [None for _ in range(max_val + 1)]
        for i, val in enumerate(prediction):
            key = mapping_rules[i]
            if out[key] is None:
                out[key] = val
            else:
                out[key] = max(out[key], val)
        out_predictions[k] = torch.tensor(out)

    return out_predictions.cuda()


def mapping(predictions: list, mapping_domains: list, attributes: list):
    for i, attribute in enumerate(attributes):
        if attribute in mapping_domains:
            if attribute == 'Main_Color':
                predictions[i] = color_mapping(predictions[i])
            else:
                sys.exit(f"No mapping for {attribute}")
    return predictions
