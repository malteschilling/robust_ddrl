from ray.rllib.models import ModelCatalog
from models.fcnet_glorot_uniform_init import FullyConnectedNetwork_GlorotUniformInitializer
from models.fcnet_glorot_normal_init import FullyConnectedNetwork_GlorotNormalInitializer

ModelCatalog.register_custom_model("fc_glorot_uniform_init", FullyConnectedNetwork_GlorotUniformInitializer)
ModelCatalog.register_custom_model("fc_glorot_normal_init", FullyConnectedNetwork_GlorotNormalInitializer)
