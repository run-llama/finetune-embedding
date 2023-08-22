# Label-Free Embedding Fine-Tuning for RAG

## Why fine-tune embeddings
Like everyone else, you build your RAG system with OpenAI embedding.
But is that the end all?
What is my data is very domain specific, with a lot of technical jargons not commonly seen on the internet.
Or the formatting of the file is very esoteric.
Or the retrieval just sucks, and you want to make it better.


You realize the generation is good. It's actually the retrieval that's failing.
You have some domain specific dataset, now you want to finetune the embedding model on it. 


## How to run this
In your environment, run `pip install -r requirements.txt` to install necessary requirements.

1. Run `generate_dataset.ipynb` to generate a synthetic dataset from a corpus of documents.
2. Run `fine_tune.ipynb` to fine-tune the embedding, and evaluate on the validation set.

That's it.

## How this works
The key idea is to use LLM to generate synthetic positive example pairs.


### Step 1: generate synthetic
> [generate_dataset.ipynb](./generate_dataset.ipynb)  

notebook for generating a synthetic dataset of question & answer pairs.

We split this into train/val split.

### Step 2: setup evaluation
> [evaluate.ipynb](./evaluate.ipynb)  
> [evaluate_st.ipynb](./evaluate_st.ipynb)  

We consider two ways of setting up the evaluator:
1. evaluator from sentence transformer: the benefit of this is that you can directly get access to a suite of evaluation
2. custom evaluator: the benefit for this is that you can evaluate arbitrary embedding models (include openAI embeddings)

### Step 3: Fine-tune your embeddings
> [fine_tune.ipyb](./fine_tune.ipynb)
Once we have our training dataset, to fine-tune is super straightforward. We just need to select a reasonable loss function, and have a good strategy for when to stop training.
