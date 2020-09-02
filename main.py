import argparse
import warnings
import pandas as pd

from simpsons_paradox import SimpsonsParadox


def main():

    pd.options.mode.chained_assignment = None
    warnings.simplefilter(action='ignore', category=UserWarning)

    parser = argparse.ArgumentParser(description="Find Simpson's Pairs")

    parser.add_argument('-input_file',
                        required=True,
                        help='input CSV file')

    parser.add_argument('-dependent_variable',
                        required=True,
                        help='dataset dependent variable')

    parser.add_argument('-model',
                        choices=['linear', 'logistic'],
                        help='type of regression to use')

    parser.add_argument('-ignore_columns',
                        default="",
                        nargs='*',
                        help='columns to ignore')

    parser.add_argument('-bin_columns',
                        default="",
                        nargs='*',
                        help='columns to bin')

    parser.add_argument('-bin_method',
                        default='quantile',
                        choices=['quantile', 'kmeans', 'uniform'],
                        help='binning method')

    parser.add_argument('-min_corr',
                        default=0.01,
                        type=float,
                        help='minimum correlation to filter pairs')

    parser.add_argument('-max_pvalue',
                        default=0.05,
                        type=float,
                        help='maximum p-value to filter pairs')

    parser.add_argument('-min_coeff',
                        default=0.00001,
                        type=float,
                        help='minimum coefficient to filter pairs')

    parser.add_argument('-standardize',
                        action='store_true',
                        help='standardize continuous variables')

    parser.add_argument('-output_plots',
                        action='store_true',
                        help='output plots and model results')

    parser.add_argument('-target_category',
                        type=int,
                        help='target category for discrete, non-binary DV')

    parser.add_argument('-weighting',
                        action='store_true',
                        help='excludes weak Simpsons pairs')

    parser.add_argument('-quiet',
                        action='store_true',
                        help='suppress warnings and verbosity')

    args = parser.parse_args()

    params = {
        'df': pd.read_csv(args.input_file),
        'dv': args.dependent_variable,
        'model': args.model,
        'ignore_columns': args.ignore_columns,
        'bin_columns': args.bin_columns,
        'bin_method': args.bin_method,
        'max_pvalue': args.max_pvalue,
        'min_coeff': args.min_coeff,
        'min_corr': args.min_corr,
        'standardize': args.standardize,
        'output_plots': args.output_plots,
        'target_category': args.target_category,
        'weighting': args.weighting,
        'quiet': args.quiet
    }

    SimpsonsParadox(**params).get_simpsons_pairs()


if __name__ == '__main__':
    main()
