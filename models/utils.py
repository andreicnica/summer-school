import torch

def load_model_weights(model, pretrained_state):
    model_state = model.state_dict()

    pretrained_weights = {}
    for k, v in pretrained_state.items():
        if k in model_state and v.size() == model_state[k].size():
            pretrained_weights[k] = v
        else:
            print(f"Not loading layer: {k}")

    model_state.update(pretrained_weights)
    model.load_state_dict(model_state)
