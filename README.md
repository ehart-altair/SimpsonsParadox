# SimpsonsParadox
Function for automatically detecting Simpson's Paradox


## Description 
This function will automatically detect Simpson’s pairs (i.e. pairs of independent and conditioning variables) in a dataset with a pre-defined DV using regression models. If the user hasn’t specified which model type to use, the function will use logistic regression if the DV is binary and one-versus-all logistic regression if the DV is but not binary. Otherwise, the function will use linear regression.  

 
This function will also do pre-processing steps prior to checking the dataset: 

- Removing user-defined irrelevant columns from the analysis 
- Encoding any non-numeric columns (i.e. string or Boolean data types) 
- Standardizing any non-discrete columns (i.e. columns with more than 10 categories) 
- Binning large conditioning variables if the user hasn’t specified any to bin 

Only pairs with pre-defined minimum correlation (between the IV and CV, and between the CV and DV) will be checked by model building. If the DV is binary, only correlation between IV and CV is checked. 

## Usage: Jupyter  
- Unzip simpsons_paradox.zip 
- Open Anaconda Prompt and run the following commands: 
- cd simpsons_paradox 
- conda env create -f environment.yml 
- conda activate simpsons-paradox 
- jupyter lab or jupyter notebook 

## Usage: Scripts  
- Unzip simpsons_paradox.zip 
- Open Anaconda Prompt and run the following commands: 
- cd simpsons_paradox 
- conda env create -f environment.yml 
- conda activate simpsons-paradox 

## More
See the documentation file for more details, including examples, and an explanation of how to use the function
