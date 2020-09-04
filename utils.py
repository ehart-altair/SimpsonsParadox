import warnings
import pandas as pd


def suppress_warnings():
    """Suppresses some warnings thrown by packages used in the function.

    Here are some possible warnings:
        - pandas's SettingWithCopyWarning: This happens when we try to
        overwrite values in a pandas series (i.e. for the one-versus-all).
        - numpy's Overflow warning: This happens when large confidence
        intervals are being computed or when probabilities of large values are
        being computed.
        - scikitlearn's KBinssDiscretizer: This happens when we're binning a
        variable that has mostly 1 value, and the function uses 'quantile'
        method so results in few bins.

    These are some of the warnings we received with our test data.
    Other warnings may occur depending on the data being used.

    Note: Keeping these warnings may help you find models with
    issues (i.e. large confidence intervals).
    """
    pd.options.mode.chained_assignment = None
    warnings.simplefilter(action="ignore", category=UserWarning)
    warnings.simplefilter(action="ignore", category=RuntimeWarning)
