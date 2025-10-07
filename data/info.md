# AC-LoRA

This folder contains the additional data required to reproduce our experiments.

Here is a summary of its content:

## RepLiQA
This folder contains the relevant data to reproduce the experiments on the [RepLiQA](https://arxiv.org/abs/2406.11811) dataset.

- db: contains the vector-database
- test: contains our test split used to evaluate AC-LoRA
- train: contains the train files (4 datapoints per question, as described in the paper) we used to finetune the LoRAs.
- db_file: contains the files we used to build the vector-store.

Most included config point to `data/RepLiQA/LoRAs/`. We do not include our finetuned LoRAs here. Please either insert your finetuned LoRAs or change the path in the config file.

## FlanV2

As mentioned in the paper, we use this [Flanv2](https://github.com/google-research/FLAN/tree/main/flan/v2) to compare AC-LoRA with related work. Mainly [LoraRetriever](https://arxiv.org/pdf/2402.09997). To this end we used their [setup](https://github.com/StyxXuan/LoraRetriever/tree/main) and their [LoRAs](https://huggingface.co/collections/Styxxxx/loraretriever-llama2-7b-loras-67247d00659f5ac3117f108c).

Before running these experiments, please first download the LoRAs.
They should at the end be stored under `data/Flan/LoRAs/`. If one wants to change the location of these, one needs to change the corresponding **config** file before running the experiments.

The Flan folder additionally contains:
- db: the vector base used by AC-LoRA to retrieve the relevant LoRAs. For space reasons it is splitted.
⚠️ Important: you need to reconstruct it into index.pkl (e.g., cat data/Flan/db/index_part_* > data/Flan/db/index.pkl) and index.faiss (e.g., cat data/Flan/db/index_flan_part_* > data/Flan/db/index.faiss)
- test: a copy of the [test file]((https://github.com/StyxXuan/LoraRetriever/blob/main/dataset/combined_test.json)) from LoraRetriever.

## WikiArts

This folder already contains everything needed to reproduce the experiments.
In case one wants to re-finetune the LoRAs one needs to download the images from [WikiArts](https://huggingface.co/datasets/huggan/wikiart)

The folder contains the following data:
- db: the vector store for wikiarts, constructed on the generated prompts.
- test: some example prompts
- train: contains the metadata of the generated prompts used to construct the db and the id of the corresponding image.

The relevant config points to `data/WikiArts/LoRAs/`. We do not include our finetuned LoRAs here. Please either insert your finetuned LoRAs or change the path in the config file.

## MMSci

This folder contains the following data we used to run the experiments on our variant of the [MMSci](https://github.com/Leezekun/MMSci) dataset:

- db: the vector store constructed using the files in train and the images from the `benchmark/train` as described from [MMSci](https://github.com/Leezekun/MMSci). These images are not required exept one wants to finetune the LoRAs or recompute the db.
- train: contains our train split used to build the db.
- test: contains the test file from `benchmark\dev\` we used to evaluate the retrival capacities. To run the experiments on needs to follow the instructions from [MMSci](https://github.com/Leezekun/MMSci) and download the images from `benchmark\dev\`.

## CS-Combi

This folder contain the generated dataset we used to evaluate the capacity from AC-LoRA to combine information.
This folder contains:
- test: contains the test set as described in the paper.
- train: contains the two train set to train the LoRAs1 and LoRAs2.
