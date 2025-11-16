from dataclasses import dataclass

from game.action import BaseActionResolver, GreedyActionResolver
from game.state import BaseStateAbstraction, IntentStateAbstraction
from models.cfr.cfr import CFR
from models.cfr.selector import CFRActionSelector
from models.gae.model import PolicyAndValueNetwork
from models.gae.selector import GAEActionSelector
from models.reinforce.model import NeuralNetworkReinforceModel, TabularReinforceModel
from models.reinforce.selector import ReinforceActionSelector
from models.selector import RandomSelector, RiskAwareSelector


@dataclass(frozen=True)
class EmptyBaselineModel:
    abstraction_cls: type[BaseStateAbstraction] = IntentStateAbstraction
    resolver_cls: type[BaseActionResolver] = GreedyActionResolver


MODEL_NAME_TO_CLS = {
    "CFR": CFR,
    "TabularReinforceModel": TabularReinforceModel,
    "NeuralNetworkReinforceModel": NeuralNetworkReinforceModel,
    "PolicyAndValueNetwork": PolicyAndValueNetwork,
    "RandomModel": EmptyBaselineModel,
    "RiskAwareModel": EmptyBaselineModel,
}

MODEL_NAME_TO_SELECTOR = {
    "CFR": CFRActionSelector,
    "TabularReinforceModel": ReinforceActionSelector,
    "NeuralNetworkReinforceModel": ReinforceActionSelector,
    "PolicyAndValueNetwork": GAEActionSelector,
    "RandomModel": RandomSelector,
    "RiskAwareModel": RiskAwareSelector,
}
