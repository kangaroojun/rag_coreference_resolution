####################
# Required Modules #
####################

# Generic/Built-in
from dataclasses import dataclass
from itertools import combinations
from queue import PriorityQueue
from typing import *
from tqdm import tqdm

# Libs
import spacy
from haystack import component
from haystack import Document

# Custom
from .coreference_models import BaseCoreferenceModel, Cluster, Mention


@dataclass
class SubDocument:
    content: str
    start_char_idx: int # inclusive
    end_char_idx: int # inclusive


@component
class LongQuestionCoreferenceAdaptation:
    def __init__(
            self, 
            coreference_model: BaseCoreferenceModel, 
            max_partition_token_size: Optional[int] = None, 
            sliding_window_token_interval: Optional[int] = 300
        ):
        self.coreference_model = coreference_model
        self.max_partition_token_size = max_partition_token_size or self.coreference_model.max_token_input_size
        self.sliding_window_token_interval = sliding_window_token_interval
        self.pos_model = spacy.load("en_core_web_sm")
        
    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """
        Get coreferences of input documents, adding representative mentions to the document content in the form of a 
        bracket after each vague mention. For example, "he" would be replaced with "he (John Doe)" if "John Doe" is 
        the representative mention for the cluster containing "he".

        Args:
            documents (List[Document]): Documents to be processed.

        Returns:
            Dict[str, List[Document]]: Dictionary containing output.
            - `documents` (List[Document]): List of processed documents.
        """
        for document in documents:
            # 1. Split input document into sub-documents (sliding window)
            sub_documents = self.get_sub_documents(document.content) # TODO: add window_move_by parameter
            
            # Get coreference clusters for each sub-document (run coreference model)
            coreference_clusters_from_sub_documents = self.get_coreference_clusters_per_sub_document(sub_documents=sub_documents)
            
            # Get all unique mentions identified across all sub-documents
            unique_mentions: Set[Mention] = self.get_unique_mentions(clusters_per_sub_document=coreference_clusters_from_sub_documents)
            
            # 2. Build weight function look-up table
            lookup = self.build_weight_function_lookup(
                sub_documents=sub_documents,
                coreference_clusters_from_sub_documents=coreference_clusters_from_sub_documents,
                unique_mentions=unique_mentions
            )
            
            # 3. Calculate distance graph
            graph, clusters = self.calculate_distance_graph(
                unique_mentions=unique_mentions,
                lookup=lookup
            )

            # 4. Get non-pronoun unique mentions
            unique_mentions_not_pronouns = self.get_non_pronoun_unique_mentions(unique_mentions=unique_mentions, full_document_text=document.content)

            # 5. Get cluster representatives
            cluster_representatives = self.get_cluster_representatives(clusters=clusters, unique_mentions_not_pronouns=unique_mentions_not_pronouns)

            # 6. Assign mentions to cluster representatives
            mentions_to_representatives = self.assign_mentions_to_cluster_representatives(clusters=clusters, cluster_representatives=cluster_representatives)

            # 7. Add representatives to mentions
            document.content = self.add_representatives_to_mentions(input_text=document.content, mentions_to_representatives=mentions_to_representatives)
            
        return {"documents": documents}
        
    def get_sub_documents(self, full_document_text: str) -> List[SubDocument]:
        """Create sub-documents from the full document text using a sliding window approach.

        Args:
            full_document_text (str): Full document text obtained via document.content.

        Returns:
            List[SubDocument]: List of sub-documents. Sub-document length is determined by self.max_partition_size, and
                the window slides by self.sliding_window_interval.
        """
        sub_documents: List[SubDocument] = []

        # Tokenize the entire text
        encoding = self.coreference_model.get_tokenizer()(
            full_document_text, 
            return_offsets_mapping=True, 
            padding=False,
            truncation=False,
        ) # TODO: enable truncation?
        char_offsets: List[Tuple[int, int]] = encoding["offset_mapping"] # List of (start, end) character positions for each token

        # Get rid of special tokens (not in the original text)
        char_offsets = [offset for offset in char_offsets if offset != (0, 0)]

        # Create sliding window sub-documents and track character indexes
        curr_token_idx = 0
        while curr_token_idx < len(char_offsets):
            # Determine the end token index of current window
            end_token_idx_exclusive = min(curr_token_idx + self.max_partition_token_size, len(char_offsets))

            # Get corresponding character indexes for the sliding window
            start_char_idx = char_offsets[curr_token_idx][0]
            end_char_idx_exclusive = char_offsets[end_token_idx_exclusive - 1][1]
            
            sub_doc = SubDocument(
                content=full_document_text[start_char_idx: end_char_idx_exclusive],
                start_char_idx=start_char_idx,
                end_char_idx=end_char_idx_exclusive - 1 # inclusive
            )

            sub_documents.append(sub_doc)
            curr_token_idx += self.sliding_window_token_interval # Slide the window
            
            if end_token_idx_exclusive == len(char_offsets):
                break
        return sub_documents
    
    def get_coreference_clusters_per_sub_document(self, sub_documents: List[SubDocument]) -> List[List[Cluster]]:
        """Get coreference clusters for each sub-document using the coreference model. Input is from the output of
        get_sub_documents.

        Args:
            sub_documents (List[SubDocument]): List of sub-documents.

        Returns:
            List[List[Cluster]]: List of coreference clusters for each sub-document.
        """
        clusters_per_sub_document: List[List[Cluster]] = []
        
        # Get clusters in each sub-document
        for sub_doc in tqdm(sub_documents, desc="Getting coreference clusters for each sub-document"):
            clusters: List[Cluster] = self.coreference_model.predict(sub_doc.content)
            for cluster in clusters:
                # Start and end indexes are currently local --> convert to global document level
                cluster.mentions = [ 
                    Mention(char_idx=(mention.char_idx[0] + sub_doc.start_char_idx , mention.char_idx[1] + sub_doc.start_char_idx), content=mention.content)
                    for mention in cluster.mentions
                ]
            clusters_per_sub_document.append(clusters)
        
        return clusters_per_sub_document
    
    def get_unique_mentions(self, clusters_per_sub_document: List[List[Cluster]]) -> Set[Mention]:
        """Get all unique mentions identified across all sub-documents. This is done by collating all mentions from all
        clusters in all sub-documents, and using a set to identify unique mentions. Input is from the output of
        get_coreference_clusters_per_sub_document.

        Args:
            clusters_per_sub_document (List[List[Cluster]]): List of coreference clusters for each sub-document.

        Returns:
            Set[Mention]: Set of unique mentions identified across all sub-documents.
        """
        unique_mentions: Set[Mention] = set()
        
        for subdocument_clusters in clusters_per_sub_document:
            for cluster in subdocument_clusters:
                unique_mentions.update(cluster.mentions) # Filters out duplicates
                
        return unique_mentions
    
    def get_unique_mentions_per_subdocument(
        self, 
        sub_documents: List[SubDocument], 
        unique_mentions: Set[Mention]
    ) -> List[List[Mention]]:
        """Get the unique mentions identified in each sub-document, including those in the sub-document but not identified
        by the model (but identified in other sub-documents). This is done by iterating through each sub-document and
        checking if each unique mention is within the sub-document based on the mention indices. Input is from the output 
        of get_sub_documents and get_unique_mentions.

        Args:
            sub_documents (List[SubDocument]): Sub-document objects that contain character indices to reference.
            unique_mentions (Set[Mention]): Set of unique mentions identified across all sub-documents.

        Returns:
            List[List[Mention]]: List of unique mentions identified in each sub-document.
        """
        
        unique_mentions_per_subdocument: List[List[Mention]] = []
        for sub_document in sub_documents:
            mentions_in_sub_document: List[Mention] = []
            for mention in unique_mentions:
                
                # Check if unique_mentions is within sub-document
                if (mention.char_idx[0] >= sub_document.start_char_idx) and (mention.char_idx[1] <= sub_document.end_char_idx):
                    # Add this unique mention to the list of unique mentions in current subdocument
                    mentions_in_sub_document.append(mention)
            
            unique_mentions_per_subdocument.append(mentions_in_sub_document) # Collate
        return unique_mentions_per_subdocument
    
    def build_weight_function_lookup(
        self, sub_documents: List[SubDocument],
        coreference_clusters_from_sub_documents: List[List[Cluster]],
        unique_mentions: Set[Mention],
    ) -> dict:
        """Build a weight function look-up table for mention pairs. 
        
        The look-up table contains the distance between each mention pair, based on the coreference clusters identified 
        in each sub-document. The distance is calculated as the number of times the mentions in the pair belong to the 
        same coreference cluster, divided by the total number of times they belong to the same cluster or different 
        clusters. As such, distance is a float value, ranging from 0 to 1. If distance > 0, the model identifies the 
        mention pair to be within the same cluster. Input is from the output of get_unique_mentions_per_subdocument and 
        get_coreference_clusters_per_sub_document.

        Args:
            sub_documents (List[SubDocument]): List of sub-documents.
            coreference_clusters_from_sub_documents (List[List[Cluster]]): List of coreference clusters for each sub-document.
            unique_mentions (Set[Mention]): Set of unique mentions identified across all sub-documents.

        Returns:
            dict: Look-up table containing the distance between each mention pair within each sub-document.
        """
        # Get the unique mentions identified in each sub-document, 
        # including those in the sub-document but not identified by the model (but identified in other sub-documents)
        unique_mentions_per_subdocument = self.get_unique_mentions_per_subdocument(sub_documents=sub_documents, unique_mentions=unique_mentions)
        
        # Initialize lookup
        lookup = dict()
            
        assert len(coreference_clusters_from_sub_documents) == len(unique_mentions_per_subdocument) # number of sub_documents
        
        # Iterate through each sub-document
        for sub_document_coreference_clusters, sub_document_unique_mentions in zip(coreference_clusters_from_sub_documents, unique_mentions_per_subdocument):
            # 1. Get all possible mention-pairs in current subdocument
            # Order does not matter - mention pairs are symmetric i.e. (ma, mb) is the same as (mb, ma)
            mention_pairs = list(combinations(sub_document_unique_mentions, 2))
            
            # Since mention pairs are symmetric, we sort the mention pairs and use the sorted pair as index for lookup table
            mention_pairs = [tuple(sorted(mention_pair)) for mention_pair in mention_pairs]
            
            # 2. For each pair, check if both mentions belong to the same coreference cluster in this sub-document
            for mention_pair in mention_pairs:                
                # Initialize d value tracking for mention-pair in lookup table (upper triangle)
                if mention_pair not in lookup.keys():
                    # Initialize value for this pair
                    lookup[mention_pair] = {
                        "s": 0,
                        "t": 0,
                    }
                
                # Check whether mention-pair is in any of the coreference clusters together
                if any([
                    (mention_pair[0] in cluster.mentions) and 
                    (mention_pair[1] in cluster.mentions)
                    for cluster in sub_document_coreference_clusters
                ]):
                    # Since both mentions in the pair belong to the same cluster in the sub-document!
                    # Increment coreference score s_i
                    lookup[mention_pair]["s"] += 1          
                else:
                    # Mentions do not belong to any cluster together in the sub-document
                    # Increment non-coreference score t_i
                    lookup[mention_pair]["t"] += 1
               
        # Calculate distance d for every pair
        for mention_pair in lookup.keys():
            lookup[mention_pair]["d"] = (lookup[mention_pair]["s"]) / (lookup[mention_pair]["s"] + lookup[mention_pair]["t"])
        
        return lookup
        
    def calculate_distance_graph(
            self, 
            unique_mentions: Set[Mention], 
            lookup: dict
        ) -> Tuple[dict, List[Cluster]]:
        """Calculate the distance graph for all mention pairs and group mentions into clusters across entire document.
        
        The distance graph is a dictionary where the key is a mention pair and the value is the distance between the pair. 
        The distance graph is obtained via Dijkstra's algorithm, where the distance between two mentions is calculated as 
        the product of the distance between the source mention and an intermediate mention, and the distance between the 
        intermediate mention and the target mention (which should be in the same cluster within the same sub-document). 
        That is to say that if mention_n is connected to mention_u with distance d_(n,u), and mention_u is connected to 
        mention_v with distance d_(u,v), then the distance between mention_n and mention_v is d_(n,u) * d_(u,v). d_(u,v) 
        is obtained via the lookup table from build_weight_function_lookup, mention_u and mention_v must fulfill the 
        condition that they are from the same cluster within the same sub-document. As we calculate the distance between 
        mention_n and all other mentions, we also keep track of the mentions that are connected to mention_n in the same 
        cluster. This allows us to group mentions that are connected to each other into clusters. The clusters are then 
        returned as a list of Cluster objects.

        Args:
            unique_mentions (Set[Mention]): Set of unique mentions identified across all sub-documents.
            lookup (dict): Look-up table containing the distance between each mention pair within each sub-document.

        Returns:
            Tuple[dict, List[Cluster]]: Tuple containing the distance graph and a list of all clusters within the document.
        """
        graph = dict() # key is the mention pair and the value is the distance between the pair
        clusters: Set[Set[Mention]] = set()
        
        for mention_n in unique_mentions:
            for mention_m in unique_mentions:
                if mention_n == mention_m:
                    graph[(mention_n, mention_m)] = 1
                else:
                    graph[(mention_n, mention_m)] = 0
            
            max_queue = self.MaxPriorityQueueForMentionPairs()
            queued = set() # To keep track of mentions that have been added to the queue
            cluster_n: Set[Mention] = set()
            
            max_queue.put(mention_pair=(mention_n, mention_n), distance=graph[(mention_n, mention_n)]) # Source node
            queued.add(mention_n)
            cluster_n.add(mention_n)

            # We are trying to get d_(n,m) for all mentions m
            while not max_queue.is_empty():
                # Extract the mention pair with the maximum distance
                _, (_, mention_u) = max_queue.extract_max()
                # Obtain all edges that are connected to mention_u
                for mention_pair in lookup.keys():
                    if mention_u in mention_pair and lookup[mention_pair]["d"] > 0:
                        mention_v = mention_pair[0] if mention_pair[1] == mention_u else mention_pair[1]
                        # Calculate potential distance from n to v
                        distance_n_to_v = graph[(mention_n, mention_u)]*lookup[tuple(sorted(mention_pair))]["d"]
                        if distance_n_to_v > graph[(mention_n, mention_v)]:
                            # Assign new value to graph if it is greater than the current value
                            graph[(mention_n, mention_v)] = distance_n_to_v
                        if mention_v not in queued:
                            max_queue.put(mention_pair=(mention_n, mention_v), distance=graph[(mention_n, mention_v)])
                            queued.add(mention_v)
                            cluster_n.add(mention_v)
            clusters.add(frozenset(cluster_n))
        clusters: List[Cluster] = [Cluster(mentions=list(cluster)) for cluster in clusters]
        return graph, clusters
    
    def get_non_pronoun_unique_mentions(
            self, 
            unique_mentions: Set[Mention], 
            full_document_text: str
        ) -> List[Mention]:
        """Get unique mentions that are not pronouns. This is done by comparing the unique mentions with the pronouns
        identified in the document using POS tagging. Input is from the output of get_unique_mentions.

        Args:
            unique_mentions (Set[Mention]): Set of unique mentions identified across all sub-documents.
            full_document_text (str): Full document text obtained via document.content.

        Returns:
            List[Mention]: List of unique mentions that are not pronouns.
        """
        if not unique_mentions or not full_document_text:
            return []
        spacy_doc = self.pos_model(full_document_text)
        unique_mentions_sorted: List[Mention] = sorted(unique_mentions)
        pronouns = [token for token in spacy_doc if token.pos_ == "PRON"] # Obtain all pronouns in the document via POS tagging
        unique_mentions_pronouns: List[Mention] = []
        pronoun_pointer = 0
        unique_mentions_pointer = 0
        while pronoun_pointer < len(pronouns) and unique_mentions_pointer < len(unique_mentions_sorted):
            pronoun = pronouns[pronoun_pointer]
            mention = unique_mentions_sorted[unique_mentions_pointer]
            # if pronoun.idx >= mention.char_idx[0] and pronoun.idx <= mention.char_idx[1]:
            #     unique_mentions_pronouns.append(mention)
            #     unique_mentions_pointer += 1
            if pronoun.text == mention.content:
                unique_mentions_pronouns.append(mention)
                unique_mentions_pointer += 1
            elif pronoun.idx < mention.char_idx[0]:
                pronoun_pointer += 1 # Pronoun we are at is before the current unique_mention
            else:
                unique_mentions_pointer += 1 # Pronoun we are at is already past the current unique_mention
        return [mention for mention in unique_mentions_sorted if mention not in unique_mentions_pronouns]
    
    def get_cluster_representatives(
            self, 
            clusters: List[Cluster], 
            unique_mentions_not_pronouns: List[Mention]
        ) -> List[str]:
        """Get the cluster representatives for each cluster. The cluster representative is the mention that appears the
        most frequently in the cluster. If there are multiple mentions with the same frequency, the first mention is
        selected. Input is from the output of get_non_pronoun_unique_mentions.

        Args:
            clusters (List[Cluster]): List of clusters.
            unique_mentions_not_pronouns (List[Mention]): List of unique mentions that are not pronouns.

        Returns:
            List[str]: List of cluster representatives.
        """
        cluster_representatives: List[str] = []
        for cluster in clusters:
            term_frequency = {}
            for mention in cluster.mentions:
                if mention in unique_mentions_not_pronouns:
                    if mention.content not in term_frequency:
                        term_frequency[mention.content] = 1
                    else:
                        term_frequency[mention.content] += 1
            cluster_representatives.append(max(term_frequency, key=term_frequency.get) if term_frequency else None)
        return cluster_representatives
    
    def assign_mentions_to_cluster_representatives(
            self, 
            clusters: List[Cluster], 
            cluster_representatives: List[str]
        ) -> Dict[Mention, str]:
        """Assign mentions to their respective cluster representatives. Input is from the output of
        get_cluster_representatives.

        Args:
            clusters (List[Cluster]): List of clusters.
            cluster_representatives (List[str]): List of cluster representatives.

        Returns:
            Dict[Mention, str]: Dictionary containing the mapping of mentions to their respective cluster representatives.
                Dictionary is sorted in descending order of mentions for replacement purposes.
        """
        representative_mention_dict = {
            mention: representative
            for cluster, representative in zip(clusters, cluster_representatives)
            for mention in cluster.mentions
        }
        mentions = sorted(representative_mention_dict.keys(), reverse=True) # Sort mentions in descending order for replacement purposes
        return {mention: representative_mention_dict[mention] for mention in mentions}
    
    def add_representatives_to_mentions(
            self, 
            input_text: str, 
            mentions_to_representatives: Dict[Mention, str]
        ) -> str:
        """Add the representative mentions to the input text. This is done by adding the representative mention in
        brackets after the mention. Input is from the output of assign_mentions_to_cluster_representatives, which
        is in descending order for easy replacement.

        Args:
            input_text (str): Input text.
            mentions_to_representatives (Dict[Mention, str]): Dictionary containing the mapping of mentions to their
                respective cluster representatives.

        Returns:
            str: Input text with the representative mentions added in brackets after the mention.
        """
        for mention, replacement in mentions_to_representatives.items():
            if mention.content == replacement or replacement is None or replacement in mention.content:
                continue # Skip if mention and replacement are the same or if replacement is None or if replacement is already in mention
            else:
                replacement_text = mention.content + " (" + replacement + ")" # Add the replacement in brackets
                new_start_idx_inclusive = mention.char_idx[0]
                new_end_idx_exclusive = mention.char_idx[1] + 1
                input_text = input_text[:new_start_idx_inclusive] + replacement_text + input_text[new_end_idx_exclusive:]
        return input_text
    
    class MaxPriorityQueueForMentionPairs:
        """
        A priority queue wrapper for mention pairs, implemented as a max-heap using Python's PriorityQueue.
        """
        def __init__(self):
            """
            Initializes the priority queue as a min-heap internally. The priorities (mention pair distances) are negated
            to simulate a max-heap.
            """
            self.min_priority_queue = PriorityQueue()
            
        def put(self, mention_pair: Tuple[Mention, Mention], distance: float):
            """
            Add a mention pair with its associated distance to the priority queue

            Args:
                mention_pair (Tuple[Mention, Mention]): Tuple of mentions
                distance (float): Distance between the mention pair obtained from the distance graph
            """
            # To turn min-heap into max-heap, simply put the negative of the value (distance)
            self.min_priority_queue.put( (-1 * distance, mention_pair))
            
        def extract_max(self):
            if not self.min_priority_queue.empty():
                # Check that queue is not empty because if it is empty, get becomes a blocking call
                negative_distance, mention_pair = self.min_priority_queue.get()
                
                # Reverse the negation and return
                distance = -1 * negative_distance
                return distance, mention_pair
            else: 
                return None
            
        def is_empty(self):
            return self.min_priority_queue.empty()