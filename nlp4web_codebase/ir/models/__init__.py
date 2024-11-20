from abc import ABC, abstractmethod
from typing import Any, Dict, Type


class BaseRetriever(ABC):

    @property
    @abstractmethod
    def index_class(self) -> Type[Any]:
        pass

    def get_term_weights(self, query: str, cid: str) -> Dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def score(self, query: str, cid: str) -> float:
        pass

    @abstractmethod
    def retrieve(self, query: str, topk: int = 10) -> Dict[str, float]:
        pass
