"""Automatic Simpson's Paradox Detector"""

from itertools import permutations
import warnings
import numpy as np
import pandas as pd

import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer


class SimpsonsParadox:
    """A class to automatically detect Simpson's Paradox in a dataset"""

    def __init__(self, df, dv, model='', ignore_columns=[], bin_columns=[],
                 bin_method='quantile', min_corr=0.01, max_pvalue=0.05,
                 min_coeff=0.00001, standardize=True, output_plots=False,
                 target_category=None, weighting=True, quiet=False):
        """
        Attributes
        ----------
        df: analysis dataset
        dv: analysis dataset's dependent variable
        model: type of regression models to build
        ignore_columns: list of variables to ignore
        bin_columns: list of variables to bin
        bin_method: method to bin variables
        min_corr: minimum correlation to check pairs
        max_pvalue: maximum p-value to filter pairs
        min_coeff: minimum coefficient to filter pairs
        standardize: option to standardize continuous variables
        output_plots: option to display plots and summary statistics
        target_category: target category for 1-versus-all regression
        weighting: option to include or exclude weak Simpson's pairs
        quiet: option to silence all warnings and verbosity

        Methods
        -------
        encode_variables():
            Encodes string and Boolean columns to integers.
        standardize_variables(columns_to_scale=[]):
            Standardizes variables with high cardinality.
        get_correlations():
            Computes correlations for numeric variables.
        get_binner(binning_method=''):
            Calls a bin function to bin large variables.
        linear_regression(dv='', iv=''):
            Builds and outputs linear regression model.
        logistic_regression(dv='', iv=''):
            Builds and outputs logistic regression model.
        build_model(dv='', iv=''):
            Builds and outputs regression model results.
        create_subgroups(iv='', cv=''):
            Disaggregates dataset and builds models.
        plot_trendline(cv='', iv='', predictions=[]):
            Plots trendlines using linear regression.
        get_simpsons_pairs():
            Checks every pair for Simpson's Paradox.

        """
        self.df = df
        self.dv = dv
        self.model = model
        self.ignore_columns = ignore_columns
        self.bin_columns = bin_columns
        self.output_plots = output_plots
        self.standardize = standardize
        self.bin_method = bin_method
        self.max_pvalue = max_pvalue
        self.min_coeff = min_coeff
        self.min_corr = min_corr
        self.target_category = target_category
        self.weighting = weighting
        self.quiet = quiet

    def encode_variables(self):
        """Encodes string or Boolean columns to integers.

        Returns:
          df: analysis dataset with encoded columns

        """
        columns_to_encode = self.df.select_dtypes(include=[
            'bool', 'object']).columns

        self.df[columns_to_encode] = pd.DataFrame(data={
            column: self.df[column].astype('category').cat.codes
            for column in columns_to_encode}, index=self.df.index)

        return self.df

    def standardize_variables(self, columns_to_scale):
        """Standardizes variables with cardinality greater than 10.

        Assumes all the variables in the dataset are numeric.

        Args:
            columns_to_scale: dataset's independent variables

        Returns:
            df: analysis dataset with standardized columns

        """
        columns_to_scale = [column for column in columns_to_scale
                            if self.df[column].nunique() > 10]

        def scale(values):
            return (values - np.mean(values)) / np.std(values)

        if columns_to_scale != []:
            scaled_columns = [column+'_scaled' for column in columns_to_scale]
            self.df[scaled_columns] = self.df[columns_to_scale].apply(scale)

        return self.df

    def get_correlations(self):
        """Computes correlations for numeric variables.

        Returns:
            corr_matrix: correlation matrix

        """
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        numeric_variables = self.df.select_dtypes(include=numerics).columns

        if len(numeric_variables) > 0:
            corr_matrix = self.df[numeric_variables].corr(method='pearson')
        else:
            corr_matrix = pd.DataFrame([])

        return corr_matrix

    @staticmethod
    def get_binner(binning_method):
        """Function to discretize a variable into 5 bins.

        The choices are: 'quantile', 'kmeans', 'uniform'

        Args:
            binning_method: string of method to use

        """
        return KBinsDiscretizer(encode='ordinal', strategy=binning_method)

    def linear_regression(self, df, iv):
        """Builds linear regression model.

        Args:
            df: dataset or subset of dataset
            iv: desired independent variable

        Returns:
            predictions: model's predictions
            output_df: model's summary table

        """

        # Fit linear regression
        X, y = df[iv], df[self.dv]
        x_ = sm.add_constant(X)
        model = sm.OLS(y, x_).fit()

        # Get coefficients and their p-values
        coefficients, pvalues = model.params, model.pvalues

        # Get model predictions
        predictions = model.predict()

        # Build model output table
        output_df = pd.DataFrame(data={
            'coefficient': coefficients, 'pvalue': pvalues
            }).reset_index()
        output_df = output_df.rename(columns={'index': 'variable'})
        output_df = output_df[output_df.variable != 'const']

        return predictions, output_df

    def logistic_regression(self, df, iv):
        """Builds logistic regression model.

        Args:
            df: dataset or subset of dataset
            iv: desired independent variable

        Returns:
            predictions: model's predictions
            output_df: model's summary table

        """

        # Fit logistic regression
        X, y = df[iv], df[self.dv]
        x_ = sm.add_constant(X)
        model = sm.Logit(y, x_).fit(method='bfgs', disp=False)

        # Get coefficients and their p-values
        coefficients, pvalues = model.params, model.pvalues

        # Get odds and their confidence intervals
        odds, odds_conf = np.exp(coefficients), np.exp(model.conf_int())
        lower_conf = odds_conf[odds_conf.index != 'const'][0].values[0]
        upper_conf = odds_conf[odds_conf.index != 'const'][1].values[0]

        # Get model predictions
        predictions = model.predict()

        # Build model output table
        output_df = pd.DataFrame(data={
            'coefficient': coefficients, 'pvalue': pvalues,
            '-95%': lower_conf, 'odds': odds, '+95%': upper_conf
            }).reset_index()
        output_df = output_df.rename(columns={'index': 'variable'})
        output_df = output_df[output_df.variable != 'const']

        return predictions, output_df

    def build_model(self, df, iv):
        """Builds single-variable regression model and stores model output.

        If user hasn't specified model type: Builds logistic regression if
        the DV is binary, one-versus-all logistic regression if the DV is
        discrete, and linear regression otherwise.

        Uses 'bfgs' for optimization to prevent singular matrix error.

        Args:
            df: dataset or subset of dataset
            iv: desired independent variable

        Returns:
            predictions: model's predictions
            output_df: model's summary table

        """
        if self.model == 'logistic' and df[self.dv].nunique() == 2:

            # Build logistic model
            predictions, output_df = self.logistic_regression(df, iv)

        elif self.model == 'logistic' and self.target_category is not None:

            # Create one-versus-all if the target category is defined
            df[self.dv] = np.where(
                df[self.dv] == self.target_category, 1, 0)

            # Build logistic model
            predictions, output_df = self.logistic_regression(df, iv)

        elif self.model == 'logistic' and 2 < df[self.dv].nunique() <= 10:
            raise ValueError('You have a non-binary DV. Pass a value to the '
                             'target_category in the function or re-bin your '
                             'DV prior to using the function.')

        elif self.model == 'linear':

            # Build linear model
            predictions, output_df = self.linear_regression(df, iv)

        else:
            # If user specified a target category
            if self.target_category is not None:

                # Create one-versus-all
                df[self.dv] = np.where(
                    df[self.dv] == self.target_category, 1, 0)

                # Build logistic model
                predictions, output_df = self.logistic_regression(df, iv)

            elif (self.target_category is None and
                  2 < df[self.dv].nunique() <= 10):
                raise ValueError('You have a non-binary DV. Pass a value to '
                                 'the target_category in the function or '
                                 're-bin your DV prior to using the function.')

            elif df[self.dv].nunique() == 2:

                # Build logistic model
                predictions, output_df = self.logistic_regression(df, iv)

            else:
                # Build linear model
                predictions, output_df = self.linear_regression(df, iv)

        return predictions, output_df

    def create_subgroups(self, iv, cv):
        """Creates summary statistics table for subgroup models.

        Builds regression models using the scaled IV,
        and stores model summaries for every subgroup.

        Args:
            iv: desired independent variable
            cv: desired conditioning variable

        Returns:
            reg_table: model's summary table for all subgroups
            coef_sign_sum: sum of coefficient signs of all subgroups
            wt_coef_sign_sum: weighted sum of all coefficient signs
            df: analysis dataset with model predictions column

        """
        coef_sign_sum, wt_coef_sign_sum = 0, 0
        bin_sizes, bin_names = [], []
        reg_table = pd.DataFrame([])

        # For each subgroup...
        for subgroup in self.df[cv].unique():
            subgroup_df = self.df[self.df[cv] == subgroup]

            # If the subgroup has unique values for its IV and DV
            if (subgroup_df[iv].nunique() > 1 and
                    subgroup_df[self.dv].nunique() > 1):

                # Build a regression model for the subgroup
                predictions, model_df = self.build_model(subgroup_df, iv)

                # Add model output to the table
                reg_table = reg_table.append(model_df)

                # Add IV coefficient sign
                coef_sign_sum += np.sign(model_df.loc[(
                    model_df['variable'] == iv), :]['coefficient'].values[0])

                # Add IV coefficient sign, weighted by subgroup size
                wt_coef_sign_sum += np.multiply(
                    np.sign(model_df.loc[(
                        model_df['variable'] == iv),
                                         :]['coefficient'].values[0]),
                    subgroup_df.shape[0])

                # Store subgroup title and size
                bin_names.append('bin '+str(subgroup))
                bin_sizes.append(subgroup_df.shape[0])

                # Store DV predictions from this IV for current subgroup
                self.df.loc[(
                    self.df[cv] == subgroup), iv+'_predictions'] = predictions

            # Add subgroup titles and sizes
            reg_table['bin_size'], reg_table['variable'] = bin_sizes, bin_names

        return reg_table, coef_sign_sum, wt_coef_sign_sum, self.df

    def plot_trendline(self, cv, iv, predictions):
        """Plots trend lines for scaled IV and predicted DV on an
        unscaled IV x-axis.

        If IV has less than 10 categories, creates x-axis ticks
        based on the number of categories.

        If the DV is binary, sets the y-axis scale at (0, 1).

        Args:
            cv: conditioning variable for disaggregated trend lines
            iv: unscaled independent variable to plot on the x-axis
            predictions: model-generated predictions using scaled IV

        """
        if not cv:

            # Plot trend line for aggregate data
            sns.regplot(x=iv, y=predictions, data=self.df, logistic=False,
                        ci=None, scatter_kws={'s': 0})

            # Set binary y-axis for binary DV
            if self.df[self.dv].nunique() == 2:
                plt.ylim(0, 1)

            # Set x-axis ticks for discrete IV
            if self.df[iv].nunique() < 10:
                plt.locator_params(axis='x', nbins=self.df[iv].nunique()-1)

            # Set plot labels
            plt.title(label="Predicting {} by {}".format(self.dv, iv))
            plt.ylabel(self.dv)

        else:

            # Sets color palette based on number of subgroups
            colors = sns.color_palette(palette='bright',
                                       n_colors=self.df[cv].nunique())

            _, ax = plt.subplots()

            # For each subgroup...
            for i, subgroup in enumerate(self.df[cv].unique()):
                subgroup_df = self.df[self.df[cv] == subgroup]

                # If the IV and DV values aren't constant
                if (subgroup_df[iv].nunique() > 1 and
                        subgroup_df[self.dv].nunique() > 1):

                    # Plot trend line for subgroup
                    sns.regplot(x=iv, y=predictions, data=subgroup_df,
                                logistic=False, ci=None, scatter_kws={'s': 0},
                                ax=ax, color=colors[i],
                                line_kws={
                                    'label': "bin {}".format(int(subgroup))})

                    # Set binary y-axis for binary DV
                    if self.df[self.dv].nunique() == 2:
                        plt.ylim(0, 1)

                    # Set x-axis ticks for discrete IV
                    if self.df[iv].nunique() < 10:
                        plt.locator_params(axis='x',
                                           nbins=self.df[iv].nunique()-1)

                    # Set plot labels
                    plt.legend(loc='lower right', prop={'size': 12})
                    plt.title("Predicting {} by {}, conditioned on {}".format(
                        self.dv, iv, cv))
                    plt.ylabel(self.dv)

        return plt.draw()

    def get_simpsons_pairs(self):
        """Checks all pairs of variables in a dataset for Simpson's Paradox.

        Uses the standardized IV for model-building and plotting trendlines,
        but keeps the unstandardized IV for the plot's x-axis scale.

        Returns:
            simpsons_pairs: a list of Simpson's Pairs

        """

        # Suppress warnings
        if self.quiet:
            warnings.simplefilter(action="ignore", category=UserWarning)
            warnings.simplefilter(action="ignore", category=RuntimeWarning)

        # Ignore columns
        if self.ignore_columns:
            self.df = self.df.drop(self.ignore_columns, axis=1)

        # Encode variables
        self.df = self.encode_variables()

        # Get continuous correlations
        corr_matrix = self.get_correlations()

        # Create variable binner
        binner = self.get_binner(self.bin_method)

        # Create candidate pairs list
        variables = [variable for variable in self.df if variable != self.dv]
        candidate_pairs = list(permutations(variables, 2))

        # Add standardized IV columns
        if self.standardize:
            self.df = self.standardize_variables(variables)

        simpsons_pairs = []

        # For each candidate pair...
        for pair in candidate_pairs:

            # Unpack pair
            iv, cv = pair[0], pair[1]

            # Use scaled IV column name if scaled
            if iv+'_scaled' in self.df:
                ind_var = iv+'_scaled'
            else:
                ind_var = iv

            # Filter out pair if little correlation
            binary_dv = self.df[self.dv].nunique() == 2
            if binary_dv:
                if all(variable in corr_matrix for variable in pair):
                    if np.abs(corr_matrix.loc[pair]) < self.min_corr:
                        continue
            else:
                if (all(variable in corr_matrix for variable in pair) and
                        self.dv in corr_matrix):
                    first_corr = corr_matrix.loc[pair]
                    second_corr = corr_matrix.loc[(cv, self.dv)]
                    if (np.abs(first_corr) < self.min_corr or
                            np.abs(second_corr) < self.min_corr):
                        continue

            # Build single-variable regression
            predictions, simple_reg = self.build_model(self.df, ind_var)
            self.df[iv+'_predictions'] = predictions

            # If user hasn't specified any columns to bin
            if self.bin_columns == []:

                # And there's many categories in the CV
                if self.df[cv].nunique() > 10:

                    # Bin the CV and build the models
                    self.df[cv+'_bin'] = binner.fit_transform(self.df[[cv]])
                    (multiple_reg,
                     coef_sign_sum,
                     wt_coef_sign_sum,
                     self.df) = self.create_subgroups(ind_var, cv+'_bin')

                # Otherwise, build the models without binning
                else:
                    (multiple_reg,
                     coef_sign_sum,
                     wt_coef_sign_sum,
                     self.df) = self.create_subgroups(ind_var, cv)
            else:
                # If user hasn't specified this CV for binning, and it's small
                if cv not in self.bin_columns and self.df[cv].nunique() <= 10:

                    # Build the models without binning
                    (multiple_reg,
                     coef_sign_sum,
                     wt_coef_sign_sum,
                     self.df) = self.create_subgroups(ind_var, cv)

                # If user hasn't specified, but it's big
                elif cv not in self.bin_columns and self.df[cv].nunique() > 10:

                    if not self.quiet:
                        # Give warning and build without binning
                        print('Warning: You are building disaggregate models '
                              'using a conditioning variable that has more '
                              'than 10 categories. Consider adding this '
                              'variable to the list of columns to bin, '
                              'or binning prior to using this function.')
                    (multiple_reg,
                     coef_sign_sum,
                     wt_coef_sign_sum,
                     self.df) = self.create_subgroups(ind_var, cv)

                # Otherwise, bin and build the models
                else:
                    self.df[cv+'_bin'] = binner.fit_transform(self.df[[cv]])
                    (multiple_reg,
                     coef_sign_sum,
                     wt_coef_sign_sum,
                     self.df) = self.create_subgroups(ind_var, cv+'_bin')

            # If all subgroups in this pair have a DV
            # or IV with constant value, skip pair
            if multiple_reg.shape[0] == 0:
                continue

            # Get coefficient and p-value from single-variable regression
            coef = simple_reg.loc[(
                simple_reg['variable'] == ind_var), :]['coefficient'].values[0]
            pvalue = simple_reg.loc[(
                simple_reg['variable'] == ind_var), :]['pvalue'].values[0]

            # Get odds confidence intervals from single-variable regression
            if binary_dv:
                lower_odds = simple_reg.loc[(
                    simple_reg['variable'] == ind_var), :]['-95%'].values[0]
                upper_odds = simple_reg.loc[(
                    simple_reg['variable'] == ind_var), :]['+95%'].values[0]
            else:
                lower_odds, upper_odds = False, False

            # Check if weighted sum of coefficient signs reverses
            if self.weighting:
                weighted_reverses = np.sign(coef) != np.sign(wt_coef_sign_sum)
            else:
                weighted_reverses = True

            # Exclude pairs with small IV coefficient
            if (np.abs(coef) > self.min_coeff or
                    (lower_odds < 1 and upper_odds > 1)):

                # Store pair if coefficient signs differ and p-value is small
                if (np.sign(coef) != np.sign(coef_sign_sum) and
                        np.sign(coef_sign_sum) != 0 and
                        pvalue < self.max_pvalue and
                        weighted_reverses):

                    simpsons_pairs.append(pair)

                    if not self.quiet:
                        print('=========================================='
                              '====================================='
                              'Warning! Simpson’s Paradox was detected in '
                              'this pair of variables: {}'.format(pair))
                        print('============================================='
                              '==================================')

                    if self.output_plots:

                        # Plot single-variable regression
                        print('Independent Variable:', iv)
                        display(simple_reg)
                        self.plot_trendline(
                            cv=None,
                            iv=iv,
                            predictions=iv+'_predictions')

                        # Plot multiple single-variable regressions
                        print('Conditioned on:', cv)
                        display(multiple_reg)
                        if cv+'_bin' in self.df:
                            self.plot_trendline(
                                cv=cv+'_bin',
                                iv=iv,
                                predictions=ind_var+'_predictions')
                        else:
                            self.plot_trendline(
                                cv=cv,
                                iv=iv,
                                predictions=ind_var+'_predictions')

                    plt.show()

        if not self.quiet:
            if simpsons_pairs == []:
                print('Congratulations! No Simpson’s Pairs were detected (e.g.'
                      ' Simpson’s Paradox was not detected for your dataset).')
            else:
                print('%d Simpson’s Pair(s) were detected in your dataset.' %
                      len(simpsons_pairs))

        return simpsons_pairs
