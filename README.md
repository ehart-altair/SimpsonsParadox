# SimpsonsParadox: Automatic Simpson's Paradox Detector


## Function Description 
This function will automatically detect Simpson’s pairs (i.e. pairs of independent and conditioning variables) in a dataset with a pre-defined DV using regression models. If the user hasn’t specified which model type to use, the function will use logistic regression if the DV is binary and one-versus-all logistic regression if the DV is discrete but not binary. Otherwise, the function will use linear regression.  

 
This function will also do pre-processing steps prior to checking the dataset: 

- Removing user-defined irrelevant columns from the analysis 
- Encoding any non-numeric columns (i.e. string or Boolean data types) 
- Standardizing any non-discrete columns (i.e. columns with more than 10 categories) 
- Binning large conditioning variables if the user hasn’t specified any to bin 

Only pairs with pre-defined minimum correlation (between the IV and CV, and between the CV and DV) will be checked by model building. If the DV is binary, only correlation between IV and CV is checked. 

## Usage: Jupyter 
1.	Unzip SimpsonsParadox-master.zip or use Git Bash to clone the repo.
2.	Open Anaconda Prompt, navigate to the directory of this project, and run the following commands:
* `conda env create -f environment.yml`
* `conda activate simpsons-paradox`
* `jupyter lab` or `jupyter notebook`
3. Modify the given notebook to run an example as follows:
```python
params = {
    'df': pd.read_csv('sample_data.csv'),
    'dv': 'sample_dv',
    'ignore_columns': ignore_columns, # list of columns to ignore
    'bin_columns': bin_columns, # list of columns to bin
    'output_plots': True,
    'standardize': True,
    'weighting': False
}

simpsons_pairs = SimpsonsParadox(**params).get_simpsons_pairs()
print(simpsons_pairs)
```
This will output a list of Simpson's pairs to the ```simpsons_pairs``` object, and display a series of plots and summary statistics tables for each pair if ```output_plots=True```.

The user can also modify various parameters (with smart defaults outlined in the [Wiki](https://github.com/ehart-altair/SimpsonsParadox/wiki)). Some of these parameters are:
```python
self.model = 'logistic 
self.ignore_columns = ['user_id']
self.bin_columns = ['timestamp']
self.output_plots = True # Displays plots and summary statistics in notebook
```

## Usage: Scripts 
You can also call this function from the command line.

1.	Unzip SimpsonsParadox-master.zip or use Git Bash to clone the repo.
2.	Open Anaconda Prompt, navigate to the directory of this project, and run the following commands:
* `conda env create -f environment.yml`
* `conda activate simpsons-paradox`
3. Refer to the [Wiki](https://github.com/ehart-altair/SimpsonsParadox/wiki) for example commands.

## Documentation
See the [Wiki](https://github.com/ehart-altair/SimpsonsParadox/wiki) for more details, including examples, argument descriptions, and an explanation of how to use the function.

## Credits
Some of the existing resources we reference in this project:
* Simpsons R Package: https://rdrr.io/cran/Simpsons/man/Simpsons.html
* Can you Trust the Trend: Discovering Simpson's Paradoxes in Social Data: https://arxiv.org/abs/1801.04385
