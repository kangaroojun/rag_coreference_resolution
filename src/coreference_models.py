####################
# Required Modules #
####################

# Generic/Built-in
from typing import *

# Libs
import functools
from fastcoref import FCoref as OriginalFCoref, CorefResult
from maverick import Maverick
from coreference_utils import Mention, Cluster, BaseCoreferenceModel     
from transformers import AutoModel, AutoTokenizer

class MaverickCoreferenceModel(BaseCoreferenceModel):
    
    hf_map: Final[Dict[str, str]] = {
        "ontonotes": "sapienzanlp/maverick-mes-ontonotes",
        "litbank": "sapienzanlp/maverick-mes-litbank",
        "preco": "sapienzanlp/maverick-mes-preco"
    }
    
    def __init__(self, model_type: Literal["ontonotes", "litbank", "preco"] = "ontonotes", device: str = "cuda"):
        self.hf_name_or_path = self.hf_map.get(model_type, "sapienzanlp/maverick-mes-ontonotes")
        self.device = device
        self.model = Maverick(
            hf_name_or_path = self.hf_name_or_path,
            device = self.device
        )
        
    def predict(self, text: str) -> List[Cluster]:
        results = self.model.predict(text) # TODO: Explore the other parameters
        clusters_char_offset: List[List[Tuple[int, int]]] = results.get("clusters_char_offsets")
        clusters_tokens: List[List[str]] = results.get("clusters_token_text")
        
        clusters: List[Cluster] = []
        for clusters_char_idx, clusters_content in zip(clusters_char_offset, clusters_tokens):
            cluster = Cluster(mentions=[])
            for mention_char_idx, mention_content in zip(clusters_char_idx, clusters_content):
                mention = Mention(char_idx=mention_char_idx, content=mention_content)
                cluster.mentions.append(mention)
            clusters.append(cluster)

        return clusters
        
    @property
    def max_token_input_size(self) -> int:
        return 24528  # Maverick uses DeBERTa Large
    
    def get_tokenizer(self) -> AutoTokenizer:
        return self.model.tokenizer
    

class FastCoreferenceModel(BaseCoreferenceModel):
    
    class PatchedFCoref(OriginalFCoref):
        def __init__(self, *args, **kwargs):
            original_from_config = AutoModel.from_config

            def patched_from_config(config, *args, **kwargs):
                kwargs['attn_implementation'] = 'eager'
                return original_from_config(config, *args, **kwargs)

            try:
                AutoModel.from_config = functools.partial(patched_from_config, attn_implementation='eager')
                super().__init__(*args, **kwargs)
            finally:
                AutoModel.from_config = original_from_config
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = self.PatchedFCoref(device=self.device)
        
    def predict(self, text: str) -> List[Cluster]:
        result: CorefResult = self.model.predict(texts=[text])[0]
        
        clusters_char_offset = result.get_clusters(as_strings=False) # End index is exclusive
        clusters_tokens = result.get_clusters()
        
        clusters: List[Cluster] = []
        for clusters_char_idx, clusters_content in zip(clusters_char_offset, clusters_tokens):
            cluster = Cluster(mentions=[])
            for mention_char_idx, mention_content in zip(clusters_char_idx, clusters_content):
                mention_char_idx = (mention_char_idx[0], mention_char_idx[1] - 1) # Make end index be inclusive
                mention = Mention(char_idx=mention_char_idx, content=mention_content)
                cluster.mentions.append(mention)
            clusters.append(cluster)

        return clusters
        
    @property
    def max_token_input_size(self) -> int:
        return 4096  # F-Coref uses Longformer
    
    def get_tokenizer(self) -> AutoTokenizer:
        raise NotImplementedError("Not yet implemented for F-Coref")