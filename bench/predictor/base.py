import abc
from dataclasses import dataclass, field

from omegaconf import DictConfig
from transformers import AutoTokenizer


@dataclass
class TestSample:
    prompt: str
    item: dict
    preds: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "item": self.item,
            "preds": self.preds
        }


class BasePredictor(abc.ABC):

    def __init__(self, model_name: str, tokenizer: AutoTokenizer, cfg: DictConfig) -> None:
        self._tokenizer = tokenizer
        self._model_name = model_name
        self._cfg = cfg

    @abc.abstractmethod
    def predict(self, samples: list[dict], sample_params: DictConfig) -> list[str]:
        raise NotImplementedError
