ALL_MODELS = {
    "mlp": MLP,
}

ALL_MODELS_BASE_TYPE = {
    "mlp": "mlp",
    "lenet": "lenet",
    "kf_mlp": "mlp"
}


def get_model(name):
    # @name         : name of the model

    assert name in ALL_MODELS, "Model %s is not on defined." % name

    return ALL_MODELS[name]
