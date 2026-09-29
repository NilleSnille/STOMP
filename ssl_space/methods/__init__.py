from ssl_space.methods.base import BaseMethod, BaseMethodMomentum

from ssl_space.methods.stomp import STOMP

METHODS = {
    # base classes
    "base": BaseMethod,
    "base_momentum": BaseMethodMomentum,
    # methods
    "stomp": STOMP,
}

__all__ = [
    "BaseMethod",
    "BaseMethodMomentum",
    
    "STOMP",
]