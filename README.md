
# Knowledge graph Retrieval-Augmented Framework for Text generation (KRAFT)

This repository contains the dataset and source code for KRAFT.

We use Langchain's Python module for
1) Prompting OpenAI's ChatGPT-3.5-Turbo model 
2) Embedding text using OpenAI's Ada-v2 text embedding model

## Setup

Run pip install -r requirements

# Dataset

We use the following 2 datasets to evaluate the performance of our framework:
- StrategyQA, which consists of Yes/No questions which need multistep logical thinking
- 2WikiMultiHop, which consists of short-answer questions which need multistep logical thinking

Both these datasets are already in the repository.

# Preprocessing

Run python src/make_mapping.py and python src/push_vectors.py

# Baseline

Giving contex_strategy parameter as None runs the baseline (no retrieval).

```
pipeline = ExperimentPipeline(
    dataset_file='datasets/2wikimultihop/queries.jsonl',
    llm=ChatOpenAI(openai_api_key="your_api_key"),
    embedding_model=OpenAIEmbeddings(openai_api_key="your_api_key"),
    contex_strategy=None,
    response_strategy=None,
    params={'choose_type': 'classic', 'choose_count': 3},
)
pipeline.wikimultihop_eval(out_file='results/wikimultihop_results.json', num_todo=50)
```

# KRAFT Experiments and Evaluation

In experiments.py, we iterative over values of k and the 2 different edge_traversal strategies.

For each, we construct an ExperimentPipeline.

#### Usage

To use the script, you will pass arguments through the command line. The available arguments are:

- `--k_values`: List of integer values for k.
- `--choose_types`: List of choose types.
- `--search_strategies`: List of search strategies.
- `--datasets`: List of dataset file paths.
- `--num_todo`: Integer representing the number of items to process from each dataset.

#### Running the Script

Here's the basic syntax for running the script from your command line:

```bash
python script_name.py --k_values [K_VALUES] --choose_types [CHOOSE_TYPES] --search_strategies [SEARCH_STRATEGIES] --datasets [DATASETS] --num_todo [NUM_TODO]
```

Replace `script_name.py` with the actual name of your script file.

#### Examples

1. **Basic Example**:
   Run the script with a single value for `k`, one choose type, one search strategy, on a single dataset processing 50 items:
   ```bash
   python script_name.py --k_values 3 --choose_types classic --search_strategies strategy1 --datasets dataset1.json --num_todo 50
   ```

2. **Multiple Parameters**:
   Run the script with multiple `k` values, choose types, and search strategies on two datasets, processing 100 items each:
   ```bash
   python script_name.py --k_values 3 5 --choose_types classic nearest_neighbor --search_strategies strategy1 strategy2 --datasets dataset1.json dataset2.json --num_todo 100
   ```

#### Output

The script will output the results to files named according to the combination of parameters and dataset names. For example, if you run the script with `k` value of 3, choose type `classic`, search strategy `strategy1`, on `dataset1.json` processing 50 items, the output file will be named `results/dataset1.json_classic_strategy1_3.json`.

#### Notes

- Ensure that the datasets are in the expected format and located in the specified paths.
- The script can handle multiple datasets and parameter combinations, but be aware that processing times will increase with more complex configurations.

# Real-time demo

Run streamlit run app.py to run the interactive demo
