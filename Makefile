# Create an empty Conda environment with Python 3.10
create_empty_conda_environment:
	conda create --name skysdsas python=3.10

# Install required packages using Conda from the 'conda-forge' channel
conda_install_packages:
	conda install -c conda-forge --yes --file requirements.txt

# Run the classification task by executing 'evaluate_model.py' script with input CSV data
run_classification_task:
	python src/ex2/evaluate_model.py data/raw/SemgHandGenderCh2.csv

# Run the time series correlation task by executing 'correlations.py' script with input pickle data
run_timeseries_correlation_task:
	python src/ex1/correlations.py data/raw/timeseries_samples.pickle
