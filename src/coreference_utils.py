from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import *
from transformers import AutoTokenizer

@dataclass(frozen=True)
class Mention:
    char_idx: Tuple[int, int]
    content: str

    def __lt__(self, other):
        if not isinstance(other, Mention):
            return NotImplemented
        return self.char_idx < other.char_idx
    
    def to_dict(self):
        return {
            "char_idx": self.char_idx,
            "content": self.content
        }
    
    @classmethod
    def from_dict(self, mention_dict):
        return Mention(
            char_idx=mention_dict["char_idx"],
            content=mention_dict["content"]
        )
    
@dataclass
class Cluster:
    mentions: List[Mention]

    def to_dict(self):
        return {
            "mentions": [mention.to_dict() for mention in self.mentions]
        }

    @classmethod
    def from_dict(self, cluster_dict):
        return Cluster(
            mentions=[Mention.from_dict(mention_dict) for mention_dict in cluster_dict["mentions"]]
        )

class BaseCoreferenceModel(ABC):
    """Abstract base class for coreference resolution models."""
    
    @abstractmethod
    def predict(self, text: str) -> List[Cluster]:
        """Predict coreference clusters for the given text."""
        pass
    
    @property
    @abstractmethod
    def max_token_input_size(self) -> int:
        """Return the maximum token input size supported by the model."""
        pass

    @abstractmethod
    def get_tokenizer(self) -> AutoTokenizer:
        """Return the tokenizer used by the model."""
        pass