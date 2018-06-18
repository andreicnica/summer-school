from .simple_net import simplenet
from .squeezenet import squeezenet1_0, squeezenet1_1
from .alexnet import alexnet

ALL_MODELS = {
    "simple-net": simple_net,
    "alexnet": alexnet,
    "squeezenet": squeezenet1_1,
    "squeezenet1.0": squeezenet1_0,
    "squeezenet1.1": squeezenet1_1,
}


def get_model(name, **kwargs):
    # @name         : name of the model

    assert name in ALL_MODELS, "Model %s is not on defined." % name

    return ALL_MODELS[name](**kwargs)
