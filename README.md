# Fine-Tuning Embedding for RAG with Synthetic Data

**UPDATE 9/10/2023**: We've included embedding finetuning abstractions into the LlamaIndex repo, so this repo is technically outdated! Please check out our [embedding fine-tuning guides](https://gpt-index.readthedocs.io/en/latest/end_to_end_tutorials/finetuning.html#finetuning-embeddings-for-better-retrieval-performance) in the core documentation.

This repo shows you how to fine-tune an embedding model to improve RAG performance even if you don't have labelled data (i.e. positive pairs of query/relevant documents). 

We walkthrough step-by-step the process of generating a synthetic dataset with LLM, finetuning an opensource embedding model, and finally evaluating the finetuned model.

We experiment with a small scale dataset of financial PDF documents, and show that finetuning the embedding model can substantially improve retrieval performance.

### Setup
To get started, clone this repo and install requirements. You also need to clone the llama_index repo to obtain the example PDFs.
```
git clone git@github.com:jerryjliu/llama_index
git clone git@github.com:run-llama/finetune-embedding.git
cd finetune-embedding
pip install -r requirements.txt
```

Then you can run the notebooks (i.e. via `jupyter lab`).
> The notebooks are fairly lightweight, and should work on almost any machines.

### Steps for running
1. Run through [generate_dataset.ipynb](./generate_dataset.ipynb) to generate a synthetic dataset for training and evaluation
2. Run through [finetune.ipynb](./finetune.ipynb) to finetune a pretrained opensource embedding model
3. Run through [evaluate.ipynb](./evaluate.ipynb) to evaluate the finetuned model against e.g. the pretrained base embedding model and proprietary OpenAI embedding model.

### How this works
**1. Generating synthetic dataset for training and evaluation**

The key idea here is that we can leverage an LLM to generate hypothetical questions that are best answered by a given piece of context. This allows us to generate synthetic positive pairs of (query, relevant documents) in a scalable way without requiring human labellers. 

More concretely, we first process the given documents into a corpus of text chunks. Then for each text chunk, we use LLM to generate a few hypothetical questions that can be answered with information from that text chunk. Finally, we collect all pairs of questions and text chunks as the dataset. 

**2. Finetuning an opensource embedding model**

We leverage the high-level model fitting API from `sentencetransformers` to very easily setup a training process. We use `MultipleNegativesRankingLoss` as the training objective and `InformationRetrievalEvaluator` as the evaluator during training. We use the opensource "BAAI/bge-small-en" as the base model and train for a small number of epochs.

**3. Evaluating the embedding model**

We compare the finetuned model against the base model, as well as the OpenAI embedding model. We evaluate with `InformationRetrievalEvaluator` as well as a simple hit rate metric.
