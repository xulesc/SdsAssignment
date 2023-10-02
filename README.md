This project implements two assignment tasks received as part of the Senior Data Scientist / Developer interview process. The project layout, environment setup and execution instructions follow.

# Project layout
```
├── Makefile :contains useful targets
├── README.md :this file
├── data
│   └── raw :contains the original data files
│       ├── SemgHandGenderCh2.csv
│       └── timeseries_samples.pickle
├── notebooks :contains exploratory notebooks
│   ├── wip_play1.ipynb
│   └── wip_play2.ipynb
├── reports :contains the report required as part of the assignment
│   └── skysds.pdf
├── requirements.txt
└── src :contains code implementing the methods developed
    ├── ex1 :code for the first assignment task
    │   ├── correlations.ipynb
    │   ├── correlations.py
    │   └── utils.py
    └── ex2 :code for the second assignment task
        ├── evaluate_model.ipynb
        ├── evaluate_model.py
        └── utils.py
```
# Installing Required Libraries

Before executing the code and notebooks bundled in this project first the required libraries must be installed. You can do this by navigating to the project folder in a terminal window and issuing the following commands.

#### If your work machine has the Make utility installed

    make create_empty_conda_environment
    conda activate skysdsas
    make conda_install_packages

#### If your work machine does not have the Make utility installed

    conda create --name skysdsas python=3.10
    conda activate skysdsas
    conda install -c conda-forge --yes --file requirements.txt

# Executing Assignment Tasks

There are two ways in which the assignment tasks can be executed locally - Commandline and Notebook. The two options have some minor differences
- For the classification task the notebook runs a 10 iterations of the evaluation in order to get an estimate of the variance of the performance metrics
- For the timeseries correlations task the notebook provides some additiona visualisations

## Command Line Option

Code for the two assignment tasks can be run from the command line by navigating to the project folder in a terminal window. There is a single runnable file for each of the two tasks.

#### If your work machine has the Make utility installed

    make run_classification_task
    make run_timeseries_correlation_task

#### If your work machine does not have the Make utility installed

    python src/ex2/evaluate_model.py data/raw/SemgHandGenderCh2.csv
    python src/ex1/correlations.py data/raw/timeseries_samples.pickle

## Notebook Option

Each assignment task can be executed through a Jupyter notebook as well. For the classification task use `src/ex2/evaluate_model.ipynb` and for the timeseries correlations task use `src/ex1/correlations.ipynb`. 

# Disclaimer

This project build and execution environment has only been tested on a single machine. The development machine was a Macbook running Ventura Mac OS with python 3.10.10 installed!



