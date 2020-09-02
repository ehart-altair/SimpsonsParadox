# SimpsonsParadox: Automatic Simpson's Paradox Detector


## Function Description 
This function will automatically detect Simpson’s pairs (i.e. pairs of independent and conditioning variables) in a dataset with a pre-defined DV using regression models. If the user hasn’t specified which model type to use, the function will use logistic regression if the DV is binary and one-versus-all logistic regression if the DV is discrete but not binary. Otherwise, the function will use linear regression.  

 
This function will also do pre-processing steps prior to checking the dataset: 

1. Removing user-defined irrelevant columns from the analysis 
2. Encoding any non-numeric columns (i.e. string or Boolean data types) 
3. Standardizing any non-discrete columns (i.e. columns with more than 10 categories) 
4. Binning large conditioning variables if the user hasn’t specified any to bin 

Only pairs with pre-defined minimum correlation (between the IV and CV, and between the CV and DV) will be checked by model building. If the DV is binary, only correlation between IV and CV is checked. 

## Usage: Jupyter 
1.	Download and unzip ``SimpsonsParadox-master.zip`` or use Git Bash to clone the repo.
2.	Open Anaconda Prompt, navigate to the directory of this project, and run the following commands:
* `conda env create -f environment.yml`
* `conda activate simpsons-paradox`
* `jupyter lab` or `jupyter notebook`
3. Modify the given notebook to run an example as follows:
```python
params = {
    'df': df, # A pandas dataframe
    'dv': 'class',
    'ignore_columns': ['user_id'], # A list of columns to ignore
    'bin_columns': ['timestamp'], # A list of columns to bin
    'output_plots': True,
    'weighting': False
}

simpsons_pairs = SimpsonsParadox(**params).get_simpsons_pairs()
```
This will output a list of Simpson's pairs to the ```simpsons_pairs``` object, and display a series of plots and summary statistics tables for each pair if ```output_plots=True```.

It's also possible to modify other parameters (with smart defaults outlined in this [Wiki page](https://github.com/ehart-altair/SimpsonsParadox/wiki/Arguments)). Some of these parameters are:
```python
self.model = 'logistic'
self.output_plots = True # Displays plots and summary statistics
self.weighting = True # Filters out weak Simpson's pairs
self.max_pvalue = 1 # Turns off p-value filtering
self.min_corr = 1 # Turns off correlation filtering
self.quiet = True # Silences all warnings and verbosity
```

In some cases, the function will run into issues if the appropriate parameters aren't selected. For example, if you try these parameters on one of the example datasets:
```python
## Example 8: COVID-19 Data
df = pd.read_csv('data/conposcovidloc.csv')
dv = 'Outcome1'
ignore_columns = ['Row_ID']

params = {
    "df": df,
    "dv": dv,
    "ignore_columns": ignore_columns,
    "output_plots": True
}

simpsons_pairs = SimpsonsParadox(**params).get_simpsons_pairs()

```
You'll get the following error:
```
ValueError: You have a non-binary DV. Pass a value to the target_category in the function or re-bin your DV prior to using the function.
```
To fix this, adjust the parameters dict by setting a target category for one-versus-all logistic regression as follows:
```python 
params = {
    "df": df,
    "dv": dv,
    "ignore_columns": ignore_columns,
    "output_plots": True,
    "target_category": 1
}
```
Now the function should be able to run successfully.
## Usage: Scripts 
You can also call this function from the command line.

1.	Download and unzip ``SimpsonsParadox-master.zip`` or use Git Bash to clone the repo.
2.	Open Anaconda Prompt, navigate to the directory of this project, and run the following commands:
* `conda env create -f environment.yml`
* `conda activate simpsons-paradox`
3. Refer to the [Wiki](https://github.com/ehart-altair/SimpsonsParadox/wiki) for example commands.

See the [Wiki](https://github.com/ehart-altair/SimpsonsParadox/wiki) for more details, including samples to run from the command line, argument descriptions and defaults, and some troubleshooting notes.

#### References
Some of the existing resources we reference in this project:
* Simpsons R Package: https://rdrr.io/cran/Simpsons/man/Simpsons.html
* Can you Trust the Trend: Discovering Simpson's Paradoxes in Social Data: https://arxiv.org/abs/1801.04385

Note: This function was created as part of a summer internship project at [Altair Engineering](https://altair.com/). Please let us know if you have any feedback and/or suggestions by starting an issue or [reaching out](mailto:walaamar@outlook.com).
