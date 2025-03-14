# Coreference Resolution

# Overview
This repository explores the [Long Question Coreference Adaptation (LQCA)](https://arxiv.org/pdf/2410.01671) paper in an attempt to provide better context to the LLM during the retrieval step. While there are models out there that can perform coreference identification (in this repository we explore [F-COREF](https://arxiv.org/pdf/2209.04280) and [Maverick](https://arxiv.org/pdf/2407.21489)), there are token limits to the input of these models. Beyond token limits, since the attention mechanism is `O(n^2)`, being able to reduce the input size is greatly helpful in improving efficiency of running these models as well.

More details of implementation decisions and findings can be found in my [documentation](./docs/documentation.md).

## **Installation**

To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

## Code Structure
- src/coreference_framework.py: Main script for running coreference framework. Can be run as a [Haystack component](https://docs.haystack.deepset.ai/docs/components) if necessary.
- src/coreference_models.py: Stores F-COREF and Maverick models.
- coreference_utils.py: Stores all the dataclasses and ABCs used for LQCA implementation.
- notebooks/: Contains example runs and Jupyter notebooks for interactive testing.
- docs/: Contains detailed documentation on dataset generation and evaluations.

## Usage Example
```python
from haystack import Document

input_text = """Alice went to the park to meet her friend Bob. She arrived early and decided to sit on a bench near the fountain. Bob, who was running late, sent her a message saying he'd be there in ten minutes. While waiting, Alice noticed a dog chasing its tail. The playful animal entertained her until Bob finally arrived. Once he reached the park, Bob apologized for the delay and handed Alice a book. The novel, which he had borrowed from the library, was one Alice had wanted to read for weeks. She thanked him and immediately began flipping through its pages. As they chatted, the dog from earlier ran up to them, wagging its tail enthusiastically. Bob remarked how energetic the dog was, and Alice agreed, saying, "I wonder if it belongs to anyone here." After spending an hour at the park, Alice and Bob decided to grab some coffee. They left the park, leaving the dog behind, and walked to the nearest café. Inside, Bob ordered a cappuccino while Alice chose a latte. The barista, noticing their cheerful conversation, smiled and handed them their drinks with a friendly comment, "Enjoy your day!" As they sipped their coffee, Alice mentioned a trip she was planning. She told Bob about the beautiful beaches she wanted to visit and how her brother had recommended the destination. Bob expressed interest in joining her if she didn’t mind. Alice laughed and said, "Let me check with my brother. He’s the one organizing everything." """

input_doc = Document(content=input_text)
output = lqca.run([input_doc])

# 'Alice went to the park to meet her (Alice) friend Bob. She (Alice) arrived early and decided to sit on a bench near the fountain. Bob, who was running late, sent her (Alice) a message saying he (Bob)\'d be there in ten minutes. While waiting, Alice noticed a dog (The playful animal) chasing its (The playful animal) tail. The playful animal entertained her (Alice) until Bob finally arrived. Once he (Bob) reached the park, Bob apologized for the delay and handed Alice a book. The novel , which he had borrowed from the library (a book)ibrary, was one Alice had wanted to read for weeks. She (Alice) thanked h (Bob)im (Bob) and immediately began flipping through its (a book) pages. As they (Alice and Bob) chatted, the dog from earlier ran up to them (Alice and Bob), wagging its (the dog) tail enthusiastically. Bob remarked how energetic the dog was, and Alice agreed, saying, "I (Alice) wonder if it (the dog) belongs to anyone here." After spending an hour at the park, Alice and Bob decided to grab some coffee. They (Alice and Bob) left the park, leaving the dog behind, and walked to the nearest café. Inside, Bob ordered a cappuccino while Alice chose a latte. The barista, noticing their (Alice and Bob) cheerful conversation, smiled and handed them (Alice and Bob) their (Alice and Bob) drinks with a friendly comment, "Enjoy your (Alice and Bob) day!" As they (Alice and Bob) sipped their (Alice and Bob) coffee, Alice mentioned a trip she (Alice) was planning. She (Alice) told Bob about the beautiful beaches she (Alice) wanted to visit and how her (Alice) brother had recommended the destination. Bob expressed interest in joining her (Alice) if she (Alice) didn’t mind. Alice laughed and said, "Let me (Alice) check with my (Alice) brother. He’s the one organizing everything." '
```
Implementation can also be observed in our [notebook](./notebooks/lqca.ipynb).

## Developer Notes

### Using a Different Coreference Model
If using a different coreference model, you would need to inherit from our [Abstract Base Class](./src/coreference_utils.py), from which you would need to ensure that it has the appropriate methods that cohere to the dataclasses we have specified.

### How Successful is LQCA?
LQCA is limited by the robustness of the coreference models utilised. While in this case we have used SOTA models, we still experience minor failures, as documented [here](./docs/documentation.md).

Additionally, due to the nature of human writing/documents, when working with excessively long documents whereby there is no overlapping window of mention of particular entities, LQCA still fails to resolve the coreference. Hence, we feel that while the premise of the paper is interesting, and it does still help provide context to the LLM during retrieval, LQCA is still limited in its effecitveness as of now.