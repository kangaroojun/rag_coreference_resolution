# The Coreference Resolution Problem

![image1.png](./images/image1.png)

This code was created to address the issue of Coreference Resolution (CR). The image above is an example of a CR issue, whereby the intention is to deconflict other terms used to refer to a particular entity, thereby providing more context to chunks during retrieval and generation phase. A simple example of this would be a document whereby "Donald Trump" is mentioned only in the introduction paragraph, before being referred to by terms such as "he", "president elect", or other terms, from which CR would make clear that these other terms still correspond to the entity "Donald Trump".

# Papers/Models Explored Initially

### [Major Entity Identification: A Generalizable Alternative to Coreference Resolution](https://arxiv.org/pdf/2406.14654)

**Date Published: 4 Oct 2024**

This paper identifies that the lack of domain generalisation in CR models can largely be attributed to differences in annotation guidelines of popular CR benchmarks, specifically annotation guidelines about what constitutes a mention. Major Entity Identification (MEI) aims to tackle the issue by providing a framework that focuses on mention clustering alone, allowing mention recognition to be conducted as a domain-specific task instead.

Presently, we do not have a mention recognition model that functions optimally, and given that the [github code](https://github.com/KawshikManikantan/MEI) within the paper was difficult to unpack, we felt that going ahead with the code proposed in this paper was not the best choice, recognising our time constraints.

### [Maverick: Efficient and Accurate Coreference Resolution Defying Recent Trends](https://arxiv.org/pdf/2407.21489)

**Date Published: 31 Jul 2024**

![image2.png](./images/image2.png)

Maverick poses an interesting contrast to the recent trend of CR developments which focus on seq2seq models, aiming to reduce parameter size via encoder-only models. It outperforms SOTA seq2seq models on the OntoNotes benchmark, and also outperforms LingMess and F-COREF on the other benchmarks, making it a viable model that can be run locally on our CPU. The model is also available on HuggingFace, with implementation detailed on their [github](https://github.com/SapienzaNLP/maverick-coref), hence this is one of the models we experimented on in our notebook.

### [Link-Append - Coreference Resolution through a seq2seq Transition-Based System](https://arxiv.org/pdf/2211.12142)

**Date Published: 22 Nov 2022**

While this model is SOTA, the large model size (13B) and high run-time even with a GPU was enough to avert our focus from this model.

### [F-COREF: Fast, Accurate and Easy to Use Coreference Resolution](https://arxiv.org/pdf/2209.04280)

F-COREF is another model that we are taking a closer look at due to its fast runtime and small model size. It is able to achieve such small model size via model distillation using LingMess as a teacher model.

### [Stanford CoreNLP](https://github.com/stanfordnlp/CoreNLP?tab=readme-ov-file)

Stanford CoreNLP is also observed as a promising avenue to tackle the CR problem. However, as it is a framework to tackle many different NLP problems, it is a hassle to install and setup, hence we chose to leave it for future exploration.

# Observations from F-COREF and Maverick

<table>
<tr>
<th>F-COREF</th>
<th>Maverick</th>
</tr>
<tr>
<td>

* [Need to adapt code](https://github.com/shon-otmazgin/fastcoref/issues/59) to work with our version of transformers
* LingMessCoref is the larger s2e model - bigger input size (Longformer: 4096 tokens) but slower and larger memory footprint. In contrast, FCoref (student model via distillation) replaces Longformer with DistilRoBERTa which is roughly 8 times faster than Longformer but has smaller input size (512 tokens). 
* Longformer uses sliding window attention which reduces attention mechanism time complexity to linear O(nw) where w is the window size. 
* Some strange clusters e.g. \['her friend Clara, who asked her what she had borrowed', 'Clara'\] (the first element is overly long)
* How to handle overlapping spans in the processing step
* There is no given 'representative value' for a given cluster (maybe an LLM processes?)
* Some failures (cannot associate "we" with "AISG")
</td>
<td>

* Similar results to fastcoref
* All Maverick models use DeBERTa-v3 (both base and large) which can model very long input texts. DeBERTa_large can handle sequences up to 24,528 tokens, making it better for long context. But this is very computationally expensive as attention mechanism incurs quadratic computational complexity (unlike Longformer, which uses sliding window attention mechanism for linear time complexity of O(nw) where w is the window size)
</td>
</tr>
</table>


For more information on the implementation of the models and their observations, take a look at the notebook within the github repo.

# Utilising Coreferences in RAG

##### Reference Paper: [Long Question Coreference Adaptation (LQCA)](https://arxiv.org/pdf/2410.01671)

Just having a model that can identify coreferences is not enough - we must be able to utilise the model in the context of RAG. The paper attempts to tackle the issue of coreference resolution in particular to **long contexts**, allowing the model to **identify and manage references effectively**.

We identify that the reason the paper focuses on long context as its problem is due to 2 main reasons:

1. For long contexts (past token limit for models), there is **no framework for coreference resolution**
2. For long contexts (within token limit for models), it is **computationally expensive**

![image3.png](./images/image3.png)

The high level idea of the paper is simple. We split the document into sub-documents, maintaining an overlapping window between adjacent sub-documents. By doing so, we hope that there is a linking effect between mentions, whereby 'Alice', linked to 'She' in the first sub-document in the example above, will also be recognised as being linked to 'her' in the later sub-document. This linking between mentions extends across the entire document, building clusters for each coreference.

![image4.png](./images/image4.png)

The LQCA method encompasses the above four steps. We slightly alter step 4, adding the representative mention in brackets instead of simply replacing the mention entirely.

### **Coreference resolution on sub-documents**

When we initialise the class, we specify the `max_partition_size` and `sliding_window_interval` , both of which are integer values that split the sub-documents at a character level. The overlapping window can be obtained by `max_partition_size` - `sliding_window_interval` .

We then run the coreference resolution models on each sub-document, obtaining the clusters within each sub-document.

### **Compute distances between mentions**

![image5.png](./images/image5.png)

If two mentions are in the same cluster in the same sub-document, we assign them an s_score of 1. If they are not in the same cluster but in the same sub-document, we assign them a t_score of 1.

For the example above, we recognise that the mention pair 'He<sub>1</sub>' and 'his<sub>1</sub>' occur in both sub-documents, but is recognised to be within the same cluster only in the first sub-document and not the second one. This means that it has an s_score of 1 and a t_score of 0.

![image6.png](./images/image6.png)

We then obtain d('He<sub>1</sub>', 'his<sub>1</sub>') = 1/2 = 0.5, whereby 0 \<= d(m<sub>a</sub>, m<sub>b</sub>) \<= 1 for all d(m<sub>a</sub>, m<sub>b</sub>). We do this for every mention pair, obtaining a lookup table of d_scores for mention pairs in the span of adjacent sub-documents (not isolated to single sub-document as mention pair can appear in multiple sub-documents due to window overlap). We then use this lookup table to obtain distances across sub-documents through another set of calculation.

![](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUfDxh7_BajZ_bfJ198-UWYs0oFNEEZmTuY78_RMbGqb2PbXS_Uae0VjAq20wwaC7dvWPiUuDUohl0LpkIn5v6k5SkCZwwj4Xhi01ZRXo0Mc3nXQF2UVWxJb2FQHnWyvrj4Nmjy5UMk0ro42pU6tvg=s2048?key=Cx4OAzwKcnBEjA4uDKeU7sVl)

The concept of the above formula is simple: If Mention A is linked to Mention C and Mention C to Mention B, Mention A is linked to Mention B. Following the formula, for the same example above, we find that:

* d(‘Tom’, ‘his<sub>2</sub>’) = d(‘Tom’, ‘his<sub>1</sub>’) \* d(‘his<sub>1</sub>’, ‘his<sub>2</sub>’) = 1 \* 1 = 1
* d(‘he<sub>1</sub>’, ‘his<sub>2</sub>’) = d(‘he<sub>1</sub>’, ‘his<sub>1</sub>’) \* d(‘his<sub>1</sub>’, ‘his<sub>2</sub>’) = 0.5 \* 1 = 0.5

![](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUcLdTt-Vco82A7U-wnOB7pZOdadNOgTHg9jXp47L2-VgWLDR_yGUOu1ZHlTfRVnEeYSZDNtFObt90aqWKwkYGkQG80GybdSYtXsx5bRYmekFYpj7XLdHJU9HKVkuLqiZh_uPtMyYeNZC_E0JGAE3vM=s2048?key=Cx4OAzwKcnBEjA4uDKeU7sVl)

We then recognise that two mentions belong to the same cluster so long as the distance between them is greater than a threshold value, k. From this, we are able to build the clusters within each document.

### **Get representative mention**

![image7.png](./images/image7.png)

The paper's method of obtaining the representative mention per cluster is to get the mention that is mentioned most often that does not contain a pronoun within each cluster. If there are multiple mentions that satisfy this condition, we select the earliest mention based on the principle of first selection.

### **Replace mentions**

![image8.png](./images/image8.png)

To prevent the possibility of context loss or any poor results due to mis-identification, we just add the mention as a bracketed string next to each original mention.