# Reproduction study: Towards Transparent and Explainable Attention Models

Code of the reproduction study of [Towards Transparent and Explainable Attention Models](https://www.aclweb.org/anthology/2020.acl-main.387/) paper (ACL 2020)

The reproduction study and this repository was part of the Fairness, Accountability, Confidentiality and Transparency in AI (FACT-AI) course teached at the Master's programme Artificial Intelligence at the Universiteit of Amsterdam (UvA) in 2021.
Collaborators of this study:
- Jeroen van Wely
- Niek IJzerman
- Caitlyn Bruys
- Jochem Soons

This codebase is based on the repository of the authors of the original paper, which can be found [here](https://github.com/akashkm99/Interpretable-Attention).

An overview of changes and additions to the original repository can be found at the end of this README.

## Installation 

Clone this repository:

```git clone git@github.com:Jeroenvanwely/FACT.git```

Move into the cloned FACT folder by running ```cd FACT```.

### Requirements 

```
torch==1.1.0
torchtext==0.4.0
pandas==0.24.2
nltk==3.4.5
tqdm==4.31.1
typing==3.6.4
numpy==1.16.2
allennlp==0.8.3
scipy==1.2.1
seaborn==0.9.0
gensim==3.7.2
spacy==2.1.3
matplotlib==3.0.3
ipython==7.4.0
scikit_learn==0.20.3
lime==0.2.0.1
```

Installing the required packages can done by creating an Anaconda environment

#### Anaconda

Create the Anaconda environment named FACT2021 by running: ```conda env create -f FACT_environment.yml ```
Subsequently, use: ```conda activate FACT2021 ``` to activate the environment with the installed prerequisites for running the code.

Now add your present working directory, in which the Transparency folder is present, to your python path: 

```export PYTHONPATH=$PYTHONPATH:$(pwd)```

To avoid having to change your python path variable each time, use: ``` echo 'PYTHONPATH=$PYTHONPATH:'$(pwd) >> ~/.bashrc``` or manually add the line above to your .bashrc file.

#### Install the English spaCy model

``` python -m spacy download en ```

## Preparing the Datasets 

Each dataset has a separate ipython notebook in the `./preprocess` folder. Follow the instructions in the ipython notebooks to download and preprocess the datasets. The datasets that were used in the original paper but were not used in the reproduction study (because they were not available for download) have been removed from this folder. 

Note that you have to run the notebooks using the python kernel within the FACT2021 conda environment (you might need to run ```pip install ipykernel``` to do this). Moreover, if you still have import problems when running the notebooks (telling you the module 'Transparency' cannot be found), you can circumvent this issue by manually adding these two lines at the top of your notebook:

```import sys```

```sys.path.append('/PATH_TO/FACT')```

Where you replace '/PATH_TO/FACT' with the path of your local machine to the FACT folder.

## Training & Running Experiments

The below mentioned commands trains a given model on a dataset and performs all the experiments mentioned in the original paper and, if wanted, the additional LIME experiment. 

### Text Classification datasets

```
python train_and_run_experiments_bc.py --dataset ${dataset_name} --data_dir . --output_dir ${output_path} --encoder ${model_name} --diversity ${diversity_weight}
```

```dataset_name``` can be any of the following: ```sst```, ```imdb```, ```yelp```,```20News_sports```.
```model_name``` can be ```vanilla_lstm```, or ```ortho_lstm```, ```diversity_lstm```. 
Only for the ```diversity_lstm``` model, the ```diversity_weight``` flag should be added. 

To also run the additional LIME experiments we included in the reproduction study, the ```run_lime``` flag should be added.
For example, to train and run experiments on the IMDB dataset with the Orthogonal LSTM including LIME experiments, use:

```
dataset_name=imdb
model_name=ortho_lstm
output_path=./experiments
python train_and_run_experiments_bc.py --dataset ${dataset_name} --data_dir . --output_dir ${output_path} --encoder ${model_name} --run_lime
```

Similarly, for the Diversity LSTM, use:

```
dataset_name=imdb
model_name=diversity_lstm
output_path=./experiments
diversity_weight=0.5
python train_and_run_experiments_bc.py --dataset ${dataset_name} --data_dir . --output_dir ${output_path} --encoder ${model_name} --diversity ${diversity_weight} --run_lime
```

### Tasks with two input sequences (NLI, Paraphrase Detection, QA)

```
python train_and_run_experiments_qa.py --dataset ${dataset_name} --data_dir . --output_dir ${output_path} --encoder ${model_name} --diversity ${diversity_weight}
```

The ```dataset_name``` can be any of ```snli```, ```qqp```, ```babi_1```, ```babi_2```, and ```babi_3```. 
As before, ```model_name``` can be ```vanilla_lstm```, ```ortho_lstm```, or ```diversity_lstm```. Note that the ```run_lime``` argument *cannot* be added to tasks that run on these datasets. 

### Additional arguments

A few additional flags can be added to run the code, of which we provide an overview:

* Run the code with a specific seed (provide int) for reproducability purposes: `--seed ${seed_value}`
* Skip training, instead load latest model (given there is one): `--skip_training`
* Skip the experiments (except rationale): `--skip_experiments`
* Skip only the rationale experiment (only applies for the text classification tasks): `--skip_rationale`
* Also plot LIME results for 1/4 and 1/2 of the test set containing the shortest quarter/half of the test instances: `--run_lime_additional`


## List of adaptations and additions to original code

Files overall:
- configurations.py: added logging of seed
- Encoder.py: adapted lines that multiplied hidden size times two which led to discrepancy between hidden size in configuration file and actual hidden size of LSTM model

Text classification framework:
- train_and_run_experiments_bc.py: added parse arguments and set_seed function
- ExperimentsBC.py: included skip arguments and call to lime experiment function
- TrainerBC.py: added lime_experiment() function to Evaluator class
- PlottingBC.py: added code to generate_graphs() function to plot LIME results
- DatasetBC.py added correct hidden size attribute to datasets
- Binary_Classification.py: added lime_analysis() function and predict_fn() function used for LIME analysis

Tasks with two input sequences:
- train_and_run_experiments_qa.py: added parse arguments and set_seed function
- ExperimentsQA.py: included skip arguments
- DatasetBC.py: added correct hidden size attribute to datasets









