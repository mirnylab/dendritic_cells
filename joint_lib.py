""" joint_lib is a technical library for supporting spectral analysis by Aleksandra Galitsyna. 


"""

import numpy as np
import pandas as pd

from cooltools.lib import numutils
from cooltools.api.eigdecomp import _filter_heatmap, _fake_cis

import scipy

import bioframe
import cooler

import pySTATIS

import tqdm

from cooltools.lib import numutils

### Plotting utils ###
import matplotlib as mpl
import matplotlib.pyplot as plt
import proplot

# Plotting parameters:
proplot.rc['figure.facecolor'] = 'white'
proplot.rc.update(
    linewidth=1, fontsize=10,
    color='dark blue', suptitlecolor='dark blue',
    titleloc='upper center', titlecolor='dark blue', titleborder=False,
)
mpl.rcParams['font.sans-serif'] = "Arial" # Set the font
mpl.rcParams['font.family'] = "sans-serif"
mpl.rcParams['grid.alpha'] = 0 # Remove the grid
mpl.rcParams['xtick.minor.bottom'] = False # Remove minor ticks
mpl.rcParams['ytick.minor.left'] = False
mpl.rcParams['ytick.minor.right'] = False


### Hi-C data operations, partially taken from and inspired by cooltools and inspectro ###

def _normalize_matrix_cis(A, perc_top=99.95, perc_bottom=1, ignore_diags=2):
    """
    Take cis Hi-C matrix, normalize by OOE, filter, zero out diagonal elements and then perform ICE.
    Filtration: _filter_heatmap from cooltools.api.eigdecomp 
    ICE: iterative_correction_symmetric from cooltools.lib.numutils

    Parameters:
    -----------
    A : 2D array
        cis Hi-C matrix.
    perc_top : float
        Percentile of top values to keep. Default: 99.95
    perc_bottom : float
        Percentile of bottom values to keep. Default: 1
    ignore_diags : int
        Number of diagonals to ignore. Default: 2

    Returns:
    --------
    OOE normalized, filtered and ICE balanced matrix

    """
    A = np.array(A)
    A[~np.isfinite(A)] = 0

    is_bad_bin = np.nansum(A, axis=0) == 0
    is_good_bin = ~is_bad_bin

    if A.shape[0] <= ignore_diags + 3 or is_good_bin.sum() <= ignore_diags + 3:
        return np.nan * np.ones(A.shape)

    if ignore_diags:
        for d in range(-ignore_diags + 1, ignore_diags):
            numutils.set_diag(A, 1.0, d)

    # Zero out bins nulled out using NaNs
    A[is_bad_bin, :] = 0
    A[:, is_bad_bin] = 0

    # Filter the heatmap
    is_valid = np.logical_and.outer(is_good_bin, is_good_bin)
    A = _filter_heatmap(A, is_valid, perc_top, perc_bottom)
    is_bad_bin = np.nansum(A, axis=0) == 0
    A[is_bad_bin, :] = 0
    A[:, is_bad_bin] = 0

    OE, _, _, _ = numutils.observed_over_expected(A, is_good_bin)

    # Inject zero diagonal, balance and rescale margins to 1
    A = numutils.set_diag(A, 0, 0)
    OE = numutils.iterative_correction_symmetric(OE)[0]
    marg = np.r_[np.sum(OE, axis=0), np.sum(OE, axis=1)]
    marg = np.mean(marg[marg > 0])
    OE /= marg

    # empty invalid rows, so that get_eig can find them
    OE[is_bad_bin, :] = 0
    OE[:, is_bad_bin] = 0

    return OE


def _normalize_matrix_trans(A, partition, perc_top=99.95, perc_bottom=1):

    """
    Take whole-genome Hi-C matrix, filter, introduce decoy cis interactions and then perform ICE.
    Filtration: _filter_heatmap from cooltools.api.eigdecomp 
    ICE: iterative_correction_symmetric from cooltools.lib.numutils

    Parameters:
    -----------
    A : 2D array
        whole-genome Hi-C matrix.
    partition : array-like
        List of partition points (in matrix index coordinates) that separate chromosomes.
    perc_top : float
        Percentile of top values to keep. Default: 99.95
    perc_bottom : float
        Percentile of bottom values to keep. Default: 1

    Returns:
    --------
    OOE normalized, filtered and ICE balanced matrix
    
    """

    A = np.array(A)
    if A.shape[0] != A.shape[1]:
        raise ValueError("A is not symmetric")

    n_bins = A.shape[0]
    if not (
        partition[0] == 0 and partition[-1] == n_bins and np.all(np.diff(partition) > 0)
    ):
        raise ValueError(
            "Not a valid partition. Must be a monotonic sequence "
            "from 0 to {}.".format(n_bins)
        )

    # Zero out cis data and create mask for trans
    extents = zip(partition[:-1], partition[1:])
    part_ids = []
    for n, (lo, hi) in enumerate(extents):
        A[lo:hi, lo:hi] = 0
        part_ids.extend([n] * (hi - lo))
    part_ids = np.array(part_ids)
    is_trans = part_ids[:, None] != part_ids[None, :]

    # Zero out bins nulled out using NaNs
    is_bad_bin = np.nansum(A, axis=0) == 0
    A[is_bad_bin, :] = 0
    A[:, is_bad_bin] = 0

    # Filter the heatmap
    is_good_bin = ~is_bad_bin
    is_valid = np.logical_and.outer(is_good_bin, is_good_bin)
    A = _filter_heatmap(A, is_trans & is_valid, perc_top, perc_bottom)
    is_bad_bin = np.nansum(A, axis=0) == 0
    A[is_bad_bin, :] = 0
    A[:, is_bad_bin] = 0

    # Inject decoy cis data, balance and rescale margins to 1
    A = _fake_cis(A, ~is_trans)
    numutils.set_diag(A, 0, 0)
    A = numutils.iterative_correction_symmetric(A)[0]
    marg = np.r_[np.sum(A, axis=0), np.sum(A, axis=1)]
    marg = np.mean(marg[marg > 0])
    A /= marg

    A = _fake_cis(A, ~is_trans)
    numutils.set_diag(A, 0, 0)
    A = numutils.iterative_correction_symmetric(A)[0]
    marg = np.r_[np.sum(A, axis=0), np.sum(A, axis=1)]
    marg = np.mean(marg[marg > 0])
    A /= marg

    A[is_bad_bin, :] = 0
    A[:, is_bad_bin] = 0

    return A


def _randomize_by_diag(A, replace=False):
    """ 
    Randomize a symmetric matrix by each diagonal.
    Randomization is performed for upper diagonal only and then mirrored to lower diagonal.

    Parameters:
    -----------
    A : 2D array
        Symmetric matrix to randomize.
    replace : bool, optional
        Whether to sample with replacement (default False).
    """
    
    n_bins = A.shape[0]
    
    # Randomize by row:
    Ar = np.zeros(A.shape)
    for i in range(n_bins):
        diag = A.diagonal(i)
        diag_randomized = np.random.choice(diag, size=len(diag), replace=replace)
        np.fill_diagonal(Ar[i:, :], diag_randomized)
    
    # Symmetrize (and do not double count diagonal):
    Ar = Ar + Ar.T - np.diag(Ar.diagonal())
    return Ar


def _normalize_matrix_cis_randomized(A, perc_top=99.95, perc_bottom=1, ignore_diags=2, replace=False):
    """
    Perform regular cis matrix normalization, but randomize the matrix by each diagonal before balancing.
    Filtration: _filter_heatmap from cooltools.api.eigdecomp 
    ICE: iterative_correction_symmetric from cooltools.lib.numutils
    Randomization: _randomize_by_diag

    Parameters:
    -----------
    A : 2D array
        cis Hi-C matrix.
    perc_top : float
        Percentile of top values to keep. Default: 99.95
    perc_bottom : float
        Percentile of bottom values to keep. Default: 1
    ignore_diags : int
        Number of diagonals to ignore. Default: 2
    replace : bool, optional
        Whether to sample with replacement (default False).

    Returns:
    --------
    OOE normalized, filtered and ICE balanced matrix

    """
    A = np.array(A)
    A[~np.isfinite(A)] = 0

    is_bad_bin = np.nansum(A, axis=0) == 0
    is_good_bin = ~is_bad_bin

    if A.shape[0] <= ignore_diags + 3 or is_good_bin.sum() <= ignore_diags + 3:
        return np.nan * np.ones(A.shape)

    if ignore_diags:
        for d in range(-ignore_diags + 1, ignore_diags):
            numutils.set_diag(A, 1.0, d)

    # Zero out bins nulled out using NaNs
    A[is_bad_bin, :] = 0
    A[:, is_bad_bin] = 0

    # Filter the heatmap
    is_valid = np.logical_and.outer(is_good_bin, is_good_bin)
    A = _filter_heatmap(A, is_valid, perc_top, perc_bottom)
    is_bad_bin = np.nansum(A, axis=0) == 0
    A[is_bad_bin, :] = 0
    A[:, is_bad_bin] = 0

    OE, _, _, _ = numutils.observed_over_expected(A, is_good_bin)

    # Randomize
    OE = _randomize_by_diag(OE, replace=False)
    
    # # Inject zero diagonal, balance and rescale margins to 1
    # numutils.set_diag(A, 0, 0)
    
    # OE = numutils.iterative_correction_symmetric(OE)[0]
    # marg = np.r_[np.sum(OE, axis=0), np.sum(OE, axis=1)]
    # marg = np.mean(marg[marg > 0])
    # OE /= marg

    # # empty invalid rows, so that get_eig can find them
    # OE[is_bad_bin, :] = 0
    # OE[:, is_bad_bin] = 0

    return OE


def retrieve_cis_matrices(
    clr,
    view_df=None,
    ignore_diags=None,
    bad_bins=None,
    balance="weight",
    perc_bottom=1,
    perc_top=99.95,
    verbose=False,
    func_retrieve=_normalize_matrix_cis,
    nthreads=1,
    **kwargs,
):
    """
    retrieve cis matrices from a cooler object.
    Parameters:
    -----------
    clr : cooler.Cooler
        Cooler object.
    view_df : DataFrame, optional
        DataFrame with regions to retrieve.
    ignore_diags : int, optional
        Number of diagonals to ignore.
    bad_bins : array-like, optional
        List of bad bins (in the coordinates of bin table index).
    balance : str, optional
        Name of the column in clr.bins() to use for balancing.
    perc_bottom : float, optional
        Percentile of bottom values to keep. Default: 1
    perc_top : float, optional
        Percentile of top values to keep. Default: 99.95
    verbose : bool, optional
        Whether to print progress. Default: False
    func_retrieve : function, optional
        Function to use for matrix retrieval. Default: _normalize_matrix_cis
    nthreads : int, optional
        Number of threads to use. Default: 1, note that nthreads > 1 is not tested.
    kwargs : dict, optional
        Additional keyword arguments to pass to func_retrieve.
    """

    # get view_df from cooler, if view_df not specified:
    if view_df is None:
        view_df = bioframe.make_viewframe(
            [(chrom, 0, clr.chromsizes[chrom]) for chrom in clr.chromnames]
        )
    else:
        # appropriate view_df checks:
        if not bioframe.is_viewframe(view_df):
            raise ValueError("view_df is not a valid view_df.")
        if not bioframe.is_contained(view_df, bioframe.make_viewframe(clr.chromsizes)):
            raise ValueError("view_df is out of the bounds of chromosomes in cooler.")

    if 'name' not in view_df.columns:
        view_df = bioframe.make_viewframe(view_df, name_style="ucsc")
    
    # ignore diags as in cooler unless specified
    ignore_diags = (
        clr._load_attrs("bins/weight").get("ignore_diags", 2)
        if ignore_diags is None
        else ignore_diags
    )

    def _each(region):
        """
        return modified matrix for a region
        Parameters
        ----------
        region: tuple-like
            tuple of the form (chroms,start,end,*)
        Returns
        -------
        _region, A -> region and matrix
        """
        _region = region[:4]  # take only (chrom, start, end, name)
        if not verbose:
            print(_region)
        A = clr.matrix(balance=balance).fetch(_region[:3])

        # filter bad_bins relevant for the _region from A
        if bad_bins is not None:
            # filter bad_bins for the _region and turn relative:
            lo, hi = clr.extent(_region)
            bad_bins_region = bad_bins[(bad_bins >= lo) & (bad_bins < hi)]
            bad_bins_region -= lo
            if len(bad_bins_region) > 0:
                # apply bad bins to symmetric matrix A:
                A[:, bad_bins_region] = np.nan
                A[bad_bins_region, :] = np.nan

        A = func_retrieve(A, perc_top=perc_top, perc_bottom=perc_bottom, ignore_diags=ignore_diags, **kwargs)

        return _region, A

    # return matrix per region (can be multiprocessed)
    if verbose:
        if nthreads > 1:
            from multiprocessing import Pool
            with Pool(2) as p:
                results = list(
                    tqdm.tqdm(p.map(_each, view_df.values), total=len(view_df.values))
                )
        else:
            results = list(
                tqdm.tqdm(map(_each, view_df.values), total=len(view_df.values))
            )
    else:
        if nthreads > 1:
            from multiprocessing import Pool
            with Pool(2) as p:
                results = list(p.map(_each, view_df.values))
        else:
            results = list(map(_each, view_df.values))

    output = {}
    for _region, A in results:
        output[f"{_region[3]}"] = A.copy()

    return output


def _view_df_to_partition(clr, view_df):
    """
    Convert view_df to partition.

    Parameters:
    -----------
    clr : cooler.Cooler
        Cooler object.
    view_df : DataFrame
        view_df with regions to retrieve.
    """
    # Ensure sorting of the chromosomes the same way as in clr:
    view_df = bioframe.sort_bedframe(view_df, bioframe.make_viewframe(clr.chromsizes))

    # create partition from view_df:
    region_last = view_df.iloc[-1, :]
    partition = np.r_[
        [clr.offset(f"{r.chrom}:{r.start}-{r.end}") for i, r in view_df.iterrows()],
        clr.extent(f"{region_last.chrom}:{region_last.start}-{region_last.end}")[1]
    ]

    return partition


def retrieve_trans_matrices(
    clr,
    view_df=None,
    bad_bins=None,
    balance="weight",
    perc_bottom=1,
    perc_top=99.95,
    func_retrieve=_normalize_matrix_trans,
    **kwargs,
):
    """
    retrieve trans matrices from a cooler object.
    Parameters:
    -----------
    clr : cooler.Cooler
        Cooler object.
    view_df : DataFrame, optional
        DataFrame with regions to retrieve.
    bad_bins : array-like, optional
        List of bad bins (in the coordinates of bin table index).
    balance : str, optional
        Name of the column in clr.bins() to use for balancing.
    perc_bottom : float, optional
        Percentile of bottom values to keep. Default: 1
    perc_top : float, optional
        Percentile of top values to keep. Default: 99.95
    func_retrieve : function, optional
        Function to use for matrix retrieval. Default: _normalize_matrix_trans
    kwargs : dict, optional
        Additional keyword arguments to pass to func_retrieve.
    """
    
    if view_df is None:
        partition = np.r_[
            [clr.offset(chrom) for chrom in clr.chromnames], len(clr.bins())
        ]
    else:
        # appropriate view_df checks:
        if not bioframe.is_view_df(view_df):
            raise ValueError("view_df is not a valid view_df.")
        if not bioframe.is_contained(view_df, bioframe.make_viewframe(clr.chromsizes)):
            raise ValueError("view_df is out of the bounds of chromosomes in cooler.")

        partition = _view_df_to_partition(clr, view_df)

    lo = partition[0]
    hi = partition[-1]

    A = clr.matrix(balance=balance)[lo:hi, lo:hi]

    if not bad_bins is None:
        A[:, bad_bins] = np.nan
        A[bad_bins, :] = np.nan

    A = func_retrieve(A, partition=partition, perc_top=perc_top, perc_bottom=perc_bottom, **kwargs)

    return A


def preload_coolers(COOLER_PATHS, CONDITIONS, BINSIZE):
    """
    Preload coolers from paths and conditions.
    Parameters:
    -----------
    COOLER_PATHS : dict
        Dictionary with paths to coolers.
    CONDITIONS : list
        List of conditions.  
    BINSIZE : int
        Binsize to use.
    """
    dict_coolers = {}
    for cond in CONDITIONS:
        path = COOLER_PATHS[cond]

        clr = cooler.Cooler(path + f"::resolutions/{BINSIZE}")

        dict_coolers[cond] = clr

    return dict_coolers


def load_cis_matrices(COOLER_DICT, 
                      CONDITIONS, 
                      view_df=None, 
                      ignore_diags=None,
                      bad_bins=None,
                      nthreads=10,
                      verbose=True,
                      **kwargs
                      ):
    """
    Load cis matrices from a dictionary of coolers.
    Parameters:
    -----------
    COOLER_DICT : dict
        Dictionary with coolers.
    CONDITIONS : list
        List of conditions.
    view_df : DataFrame, optional
        DataFrame with regions to retrieve.
    ignore_diags : int, optional
        Number of diagonals to ignore.
    bad_bins : array-like, optional
        List of bad bins (in the coordinates of bin table index, or a vector).
    nthreads : int, optional
        Number of threads to use. Default: 10
    verbose : bool, optional
        Whether to print progress. Default: True
    kwargs : dict, optional
        Additional keyword arguments to retrieve_trans_matrices
    """
    OUTPUT = {}
    for cond in (pbar := tqdm.tqdm(CONDITIONS)):
        pbar.set_postfix_str(cond)

        clr = COOLER_DICT[cond]

        if view_df is None:
            view_df = bioframe.make_viewframe(clr.chromsizes)

        if bad_bins is not None:
            if bad_bins.dtype=='bool':
                bad_bins = np.where(bad_bins)[0]
            
        OUTPUT[cond] = dict(retrieve_cis_matrices(
            clr=clr,
            view_df=view_df,
            bad_bins=bad_bins,
            verbose=verbose,
            ignore_diags=ignore_diags,
            nthreads=nthreads,
            **kwargs
        ))

    return OUTPUT


def load_trans_matrices(COOLER_DICT, 
                        CONDITIONS, 
                        view_df=None,
                        bad_bins=None,
                        **kwargs
                        ):
    """
    Load trans matrices from a dictionary of coolers.
    Parameters:
    -----------
    COOLER_DICT : dict
        Dictionary with coolers.
    CONDITIONS : list
        List of conditions.
    view_df : DataFrame, optional
        DataFrame with regions to retrieve.
    bad_bins : array-like, optional
        List of bad bins (in the coordinates of bin table index, or a vector).
    kwargs : dict, optional
        Additional keyword arguments to retrieve_trans_matrices
    """

    if view_df is None:
        view_df = bioframe.make_viewframe(clr.chromsizes)
            
    if bad_bins is not None:
        if bad_bins.dtype=='bool':
            bad_bins = np.where(bad_bins)[0]
            
    OUTPUT = {}

    for cond in (pbar := tqdm.tqdm(CONDITIONS)):
        pbar.set_postfix_str(cond)

        clr = COOLER_DICT[cond]

        OUTPUT[cond] = retrieve_trans_matrices(
            clr=clr,
            view_df=view_df,
            bad_bins=bad_bins,
            **kwargs
        )

    return OUTPUT


def load_statis(
    A_dict,
    bad_bins=None,
    conditions=None,
    norm=["norm_one"],
    verbose=True,
):
    """
    Load STATISData objects from a dictionary of matrices.
    Parameters:
    -----------
    A_dict : dict
        Dictionary with matrices (normalized; each key is condition, each element is matrix).
    bad_bins : array-like, optional
        List of bad bins (in the coordinates of bin table index, or a vector).
    conditions : list, optional
        List of conditions to use. Default: None (all of them)
    norm : str, optional
        Normalization method to use. Default: "norm_one" (alternatives: None, 'zscore', 'double_center')
    verbose : bool, optional
        Whether to print ratios of zeros before and after filtration by bad_bins. Default: True

    Returns:
    --------
    X : list
        List of STATISData objects.
    good_bins : array
        Array of good bins.    
    """
    if conditions is None:
        conditions = list(A_dict.keys())

    As = np.array([A_dict[cond] for cond in conditions])

    # Mask bins with zero sum:
    good_bins = np.all(As.sum(axis=1) != 0, axis=0)

    # Mask adiitional user-provided bad bins:
    if not bad_bins is None:
        good_bins[bad_bins] = False

    X = []
    for i, cond in enumerate(conditions):

        A = A_dict[cond][good_bins, :][:, good_bins]
        
        A_statis = pySTATIS.STATISData(A, cond, normalize=norm)

        X.append(A_statis)

        prc_zeros_init = (
            100
            * np.sum(As[i, :, :] == 0)
            / np.float(As[i, :, :].shape[0] * As[i, :, :].shape[1])
        )
        prc_zeros_result = (
            100 * np.sum(A == 0) / np.float(A.shape[0] * A.shape[1])
        )
        print(
            f"""Loading into STATIS objects: {cond}
        zeros ratio before zero rows removal: {prc_zeros_init:.2f}% 
        zeros ratio after zero rows removal: {prc_zeros_result:.2f}%
        """
        )

    return X, good_bins


def run_statis(X, n_comp):
    """
    Run STATIS on a list of STATISData objects.
    Parameters:
    -----------
    X : list
        List of STATISData objects.
    n_comp : int
        Number of components to extract.

    Returns:
    --------
    statis_model : pySTATIS.STATIS
        STATIS object.
    """
    statis_model = pySTATIS.STATIS(n_comps=n_comp)
    statis_model.fit(X)

    return statis_model


def summary_statis(statis_model, CONDITIONS, n_comp_plot=10, COMP_TARGET=1, REGION=(0, 1000)):

    # Import additional visualization libraries:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns
    import proplot

    # Plotting parameters:
    proplot.rc["figure.facecolor"] = "white"
    proplot.rc.update(
        linewidth=1,
        fontsize=10,
        color="dark blue",
        suptitlecolor="dark blue",
        titleloc="upper center",
        titlecolor="dark blue",
        titleborder=False,
    )
    mpl.rcParams["font.sans-serif"] = "Arial"  # Set the font
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["grid.alpha"] = 0  # Remove the grid
    mpl.rcParams["xtick.minor.bottom"] = False  # Remove minor ticks
    mpl.rcParams["ytick.minor.left"] = False
    mpl.rcParams["ytick.minor.right"] = False

    import warnings

    warnings.filterwarnings("ignore")

    n_comp = statis_model.D_.shape[0]
    NBINS = statis_model.factor_scores_.shape[0]

    print("Eigenvalues (D) plot")
    plt.figure(figsize=[10, 5])
    sns.barplot(y=statis_model.D_, x=np.arange(n_comp), color="grey")  # list of eigenvalues
    plt.xticks(np.arange(0, n_comp, 5), rotation=90)
    plt.title("Eigenvalues")
    plt.show()

    print("Percentage of variance explained:")
    print((100 * np.power(statis_model.D_, 2) / sum(np.power(statis_model.D_, 2)))[:n_comp_plot])

    # statis_model.print_variance_explained()

    print(
        "Weights of each dataset, assessing the contribution of each dataset to the compromise"
    )
    print(statis_model.table_weights_)

    print("Barplot of weights")
    plt.figure(figsize=[10, 5])
    sns.barplot(y=statis_model.table_weights_, x=CONDITIONS, color="grey")
    plt.title("Weights of the datasets, a")
    plt.tight_layout()
    plt.show()

    # What datasets determined the components?
    print("Heatmap of contributions of datasets to components")
    plt.figure(figsize=[20, 5])
    sns.heatmap(statis_model.contrib_dat_[:, :n_comp_plot], square=True, cmap="Reds")
    plt.yticks(np.arange(0, len(CONDITIONS)) + 0.5, CONDITIONS, rotation=0)
    plt.ylabel("Dataset")
    plt.xlabel("Component")
    plt.title("Heatmap of contributions of datasets to the components:")
    plt.tight_layout()
    plt.show()

    # What genomic positions determined the components?
    print("Contributions of genomic positions per component")
    plt.figure(figsize=[20, 3])
    for i in range(n_comp_plot):
        plt.plot(statis_model.contrib_obs_[REGION[0] : REGION[1], i], label=f"PC{i}")
    plt.xlim(0, REGION[1] - REGION[0])
    plt.legend()
    plt.title("Contribution of genomic positions (observations) to the components:")
    plt.tight_layout()

    # Let's take PC1 and see what are the contributions of different datasets:
    print("Contribution of genomic positions per dataset into component 1")
    plt.figure(figsize=[20, 3])
    i = COMP_TARGET  # Number of the component
    for k, cond in enumerate(CONDITIONS):
        plt.plot(
            statis_model.contrib_var_[k * NBINS : (k + 1) * NBINS, i], label=f"PC{i} for {cond}"
        )
    plt.xlim(0, 1_000)  # NBINS)
    plt.legend()
    plt.title("Contribution of genomic positions (observations) to the components:")
    plt.tight_layout()

    print("Quick scatterplots of loadings")
    Q = statis_model.Q_
    hi = COMP_TARGET  # PC1
    i2 = COMP_TARGET + 1  # PC2

    N = len(CONDITIONS)
    n_cols = 3
    n_rows = int(np.ceil(N // 3)) + 1

    fig, axes = plt.subplots(
        n_rows, n_cols, squeeze=False, figsize=[3 * n_cols, 3 * n_rows]
    )
    axes = axes.flatten()
    for k, cond in enumerate(CONDITIONS):
        ax = axes[k]
        sns.scatterplot(
            x=Q[k * NBINS : (k + 1) * NBINS, hi],
            y=Q[k * NBINS : (k + 1) * NBINS, i2],
            s=3,
            alpha=0.7,
            hue=np.arange(NBINS),
            palette="spectral",
            ax=ax,
        )
        # plt.colorbar(points)
        ax.get_legend().remove()
        ax.set_title(cond)
        ax.set_facecolor("black")

    fig.tight_layout()
    fig.suptitle("Loadings", y=0.999)
    fig.show()

    print("Quick scatterplots of factor scores")
    F = statis_model.factor_scores_
    Fp = statis_model.partial_factor_scores_

    hi = COMP_TARGET  # PC1
    i2 = COMP_TARGET + 1  # PC2

    fig, axes = plt.subplots(
        n_rows + 1, n_cols, squeeze=False, figsize=[3 * n_cols, 3 * (n_rows + 1)]
    )
    axes = axes.flatten()

    for k, cond in enumerate(CONDITIONS):
        ax = axes[k]
        points = sns.scatterplot(
            x=Fp[k, :, hi],
            y=Fp[k, :, i2],
            s=3,
            alpha=0.7,
            hue=np.arange(NBINS),
            palette="spectral",
            ax=ax,
        )
        # plt.colorbar(points)
        ax.get_legend().remove()
        ax.set_title(cond)
        ax.set_facecolor("black")

    ax = axes[-1]
    points = sns.scatterplot(
        x=F[:, hi],
        y=F[:, i2],
        s=3,
        alpha=0.7,
        hue=np.arange(NBINS),
        palette="spectral",
        ax=ax,
    )
    ax.set_title("Factor scores for the compromise")
    ax.get_legend().remove()
    ax.set_facecolor("black")

    fig.tight_layout()
    fig.suptitle("Factor Scores with Compromise", y=0.999)
    fig.show()


def _phase_track(vec, phasing_track, method="spearman"):
    """
    Phase a vector using a phasing track.
    Parameters:
    -----------
    vec : array-like
        Vector to phase.
    phasing_track : array-like
        Phasing track.
    method : str, optional
        Method to use for phasing. Default: "spearman", "pearson" is also available.
    """
    mask = np.isfinite(phasing_track) & np.isfinite(vec)
    if method=="spearman":
        corr = scipy.stats.spearmanr(vec[mask], phasing_track[mask])[0]
    elif method=="pearson":
        corr = scipy.stats.pearsonr(vec[mask], phasing_track[mask])[0]
    else:
        raise ValueError(f"Unknown method: {method}")
    return np.sign(corr) * vec

def parse_statis_output(statis_model, 
                        bins, 
                        conditions, 
                        good_bins=None, 
                        phasing_track=None,
                        phasing_method=None,
                        n_comp=10, 
                        postfix=""):
    """ 
    Add columns to the bin table corresponding to GSVD output (STATIS-type):
    - loadings
    - loadings normalized by square of singular values
    - compromise (consensus dimensions)
    - factor scores (partial factor scores for each dataset)

    Optionally, add postfix to each parsed column output.

    Parameters:
    -----------
    statis_model : pySTATIS.STATIS
        STATIS object.
    bins : DataFrame
        Bin table.
    conditions : list
        List of conditions.
    good_bins : array-like, optional
        List of good bins (in the coordinates of bin table index).
    phasing_track : DataFrame, optional
        Phasing track DataFrame (dimension is equal to total number of bins in bin table)
        It will be used for flipping the sign of the loadings, consensus and factor scores.
    n_comp : int, optional  
        Number of components to parse. Default: 10
    postfix : str, optional 
        Postfix to add to each parsed column. Default: ""
    """
    bins = bins.copy()

    if n_comp is None:
        n_comp = statis_model.D_.shape[0]
    else:
        n_comp = n_comp
    NBINS = statis_model.factor_scores_.shape[0]

    bins.loc[:, "is_bin_valid"] = False
    bins.loc[good_bins, "is_bin_valid"] = True

    assert NBINS == len(bins.query("is_bin_valid==True"))

    # Loadings:
    for k, cond in enumerate(conditions):
        for i in range(n_comp):
            loading = statis_model.Q_[k * NBINS : (k + 1) * NBINS, i]
            if phasing_method is not None:
                loading = _phase_track(loading, phasing_track[good_bins], method=phasing_method)
            bins.loc[good_bins, f"loading:{cond}:{i}"+postfix] = loading

    # Loadings normalized by square of singular values:
    for k, cond in enumerate(conditions):
        for i in range(n_comp):
            loading = statis_model.Q_[k * NBINS : (k + 1) * NBINS, i]
            if phasing_method is not None:
                loading = _phase_track(loading, phasing_track[good_bins], method=phasing_method)
            sv = statis_model.D_[i]
            bins.loc[good_bins, f"loading-norm:{cond}:{i}"+postfix] = loading * sv**2

    # Compromise:
    for i in range(n_comp):
        dimension = statis_model.factor_scores_[:, i]
        if phasing_method is not None:
                dimension = _phase_track(dimension, phasing_track[good_bins], method=phasing_method)
        bins.loc[good_bins, f"consensus-dimension:{i}"+postfix] = dimension

    # Factor scores:
    for k, cond in enumerate(conditions):
        for i in range(n_comp):
            dimension = statis_model.partial_factor_scores_[k, :, i]
            if phasing_method is not None:
                dimension = _phase_track(dimension, phasing_track[good_bins], method=phasing_method)
            bins.loc[good_bins, f"factor-scores:{cond}:{i}"+postfix] = dimension

    variance_explained = np.power(statis_model.D_, 2) / sum(np.power(statis_model.D_, 2))
    return bins, variance_explained


def project(A, reference, n_comp=None, normalize=True):
    """
    Project matrix A to the reference list of eigenvectors or factor scores.
    Adapted from:
    https://github.com/open2c/inspectro/blob/94c3e1897f6a709b843f3604bb97dfa32e6cd3db/inspectro/utils/eigdecomp.py#L541

    A : Input matrix
    reference : eigenvector or factor scores DataFrame or matrix
    n_comp : limiting number of eigenvectors/factor scores
    normalize : whether to normalize the projected values

    Returns
    -------
    proj_table -> DataFrame (n, n_comp + 1)
        Table of projected values (as columns).

    """

    # Input check:
    n_bins = A.shape[0]
    if n_bins != len(reference):
        raise ValueError(
            f"Matrix and reference shape mismatch: {n_bins} {len(reference)}"
        )

    # Filter out missing data:
    if not isinstance(reference, np.ndarray):
        R = reference.to_numpy()
    R = reference.astype(np.float64)
    mask = (np.sum(np.abs(A), axis=0) != 0) & (np.sum(np.isnan(R), axis=1) == 0)
    A_collapsed = A[mask, :][:, mask].astype(float, copy=True)
    R_collapsed = R[mask, :]

    # Project:
    proj = np.full((n_bins, n_comp + 1), np.nan)
    result = []
    for i in range(n_comp + 1):
        result.append(np.dot(A_collapsed, R_collapsed[:, i][:, np.newaxis]))
    proj[mask, :] = np.hstack(result)

    # Normalize:
    if normalize:
        for i in range(n_comp + 1):
            d = proj[:, i]
            proj[:, i] /= np.linalg.norm(d[np.isfinite(d)])

    return proj


def transform(X, reference, good_bins=None, n_comp=None, whiten=True, variance_explained=None):
    """
    Sklearn-inspired transformation:
    https://github.com/scikit-learn/scikit-learn/blob/3f89022fa04d293152f1d32fbc2a5bdaaf2df364/sklearn/decomposition/_base.py#L101

    X is projected on the first principal components previously extracted
        from a training set (reference, can be obtained by any other method).

    Performs re-scaling to the square root of the variance explained if whiten.

    """
    # if mean_ is not None:
    #     X = X - mean_

    if good_bins is None:
        good_bins = (np.sum(np.abs(X), axis=0) != 0) & (
            np.sum(np.isnan(reference), axis=1) == 0
        )
    n_bins = X.shape[0]
    assert n_bins == len(good_bins)

    X_collapsed = X[good_bins, :][:, good_bins].astype(float, copy=True)
    ref_collapsed = reference[good_bins, :]

    if n_comp is None:
        n_comp = reference.shape[1]

    X_transformed = np.dot(X_collapsed, ref_collapsed)
    if whiten:
        if variance_explained is None:
            raise ValueError("Provide variance exmplained if requesting whiten")
        X_transformed /= np.sqrt(variance_explained[:n_comp])

    X_t_full = np.full((n_bins, n_comp), np.nan)
    X_t_full[good_bins, :] = X_transformed

    return X_t_full


def expand(eigs, factor, interpolate=None):
    """
    Expand the track of eigenvectors by a given factor.
    """

    columns = eigs.columns
    eigs = eigs.values
    eigs = np.repeat(eigs, factor, axis=0)

    if not interpolate is None:
        eigs = interpolate(eigs)

    return pd.DataFrame(eigs, columns=columns)


def imshow_rasterized(x, func=np.nanmean, ax=None, coarse_factor=10, **kwargs):
    """
    Plot rasterized linear heatmap
    """
    if len(x.shape) == 1:
        x = np.array([x])

    X = numutils.coarsen(func, x, {1: coarse_factor}, trim_excess=True)
    if ax is None:
        ax = plt
    ax.matshow(
        X,
        rasterized=True,
        aspect="auto",
        # extent=extent,
        **kwargs,
    )



#############################################
#### Plotting utils for Sankey diagram: #####
#############################################

def groups2sankey_multicolor(df, cols_states, n_states, n_clusters, cmap):
    """ More advanced implementation of df_links that sorts the links by nodes appearance"""
    clusters_numbered = np.arange(n_clusters)
    colors_clusters = [ mpl.colors.rgb2hex(x) for x in cmap.colors[:n_clusters]]
    colors_clusters_alpha = [ "rgba("+",".join( [str(x) for x in ((*mpl.colors.hex2color( mpl.colors.rgb2hex(x) ), 0.2))] )+")" for x in cmap.colors[:n_clusters]]
    
    df_nodes = pd.DataFrame({
        'cluster': list(clusters_numbered) * n_states, 
        'color': list(colors_clusters) * n_states,
        'cond': np.repeat(cols_states, n_clusters),
        'x' : np.repeat(range(n_states), n_clusters)*10
    })
    
    df_links = df.copy()

    for i in range(n_states):
        df_links.loc[:, "true" + cols_states[i]] = df_links.loc[:, cols_states[i]].astype(int)
        df_links.loc[:, f"color {cols_states[i]}"] = [colors_clusters_alpha[i] for i in df_links.loc[:, cols_states[i]].astype(int).values ]
        df_links.loc[:, cols_states[i]] = df_links.loc[:, cols_states[i]].astype(int) + i*(n_clusters)

    df_links.loc[:, 'color_left']  = df_links.apply(lambda x: x[f"color {cols_states[0]}"], axis=1)
    df_links.loc[:, 'color_right'] = df_links.apply(lambda x: x[f"color {cols_states[-1]}"], axis=1)

    dfs = []
    for state1, state2 in zip(cols_states[:-1], cols_states[1:]):
        df_tmp = df_links.loc[:, [state1, state2, 'nbins', 'color_left']+list( df_links.columns )]
        df_tmp.columns=['left', 'right', 'value', 'color']+list( df_links.columns )
        
        dfs.append(df_tmp.copy())

    df_links = pd.concat(dfs, axis=0).sort_values(["true" + state for state in cols_states])
    
    return df_nodes, df_links


def add_alpha(color, alpha):
    color = mpl.colors.hex2color(color)
    color = "rgba("+",".join( [str(x) for x in ((*color, alpha))] )+")"
    return color


def group2sankey(dataframe: pd.DataFrame, 
                    cols_grouping, 
                    col_counts,
                    colors_nodes = None,
                    order_nodes = None,
                    rename_nodes_rule = None,
                    links_colors_fetcher = None,
                    ):
    """
    Convert dataframe to sankey diagram format.
    Parameters:
    -----------
    dataframe : DataFrame
        DataFrame with groups.
    cols_grouping : list
        List of columns to use for grouping.
    col_counts : str
        Column with counts.
    colors_nodes : dict, list, optional
        Colors of nodes. Default: None (black).
    order_nodes : list, optional
        Order of nodes. Default: None (unique values in each column).
    rename_nodes_rule : dict, optional
        Rule to rename nodes. Default: None.
    links_colors_fetcher : function, optional
        Function to fetch colors of links. Default: None (black).
    """

    n_groups = len(cols_grouping)

    if order_nodes is None:
        # Create order out of unique value in each column that is supposed to be a group:
        order_nodes = [dataframe[col].unique() for col in cols_grouping]
    else:
        # Validate that then order is correct:
        for i, order in enumerate(order_nodes):
            col = cols_grouping[i]
            assert set(dataframe[col].unique())==set(order)

    len_groups = [len(order_nodes[i]) for i in range(n_groups)]

    # Define nodes:
    nodes = [node for i in range(n_groups) for node in order_nodes[i]]
    
    # Define colors of nodes:
    if colors_nodes is None:
        colors = np.repeat('black', len(nodes))
    elif type(colors_nodes) is dict:
        colors = [colors_nodes[node] for node in nodes]
    elif type(colors_nodes) is list:
        colors = [colors_nodes[j] for i in range(n_groups) for j in range(len(order_nodes[i]))]
    else:
        raise ValueError(f"Unknown format for colors_nodes, or not implemented: {colors_nodes}")
    
    # Define conditions (column origins):
    cols = np.concatenate([np.repeat(cols_grouping[i], len_groups[i]) for i in range(n_groups)])

    # Define x position of nodes:
    xs = np.concatenate([np.repeat(range(n_groups)[i], len_groups[i]) for i in range(n_groups)])*10

    # Create node dataframe
    df_nodes = pd.DataFrame({
        'node': nodes, 
        'color': colors,
        'col': cols,
        'x' : xs
    }).reset_index()
    if rename_nodes_rule is not None:
        df_nodes.loc[:, 'node_name'] = df_nodes['node'].apply(lambda x: rename_nodes_rule[x])

    # Define links:
    df_links = dataframe[cols_grouping+[col_counts]].copy()
    for i, col in enumerate(cols_grouping):
        node2index = dict( df_nodes.loc[df_nodes['col']==col, :][['node', 'index']].values )
        df_links.loc[:, f'col{i}'] = df_links[col].apply(lambda x : node2index[x])

    if links_colors_fetcher is None:
        df_links.loc[:, 'color'] = 'black'
    else:
        df_links.loc[:, 'color'] = df_links.apply(lambda x: links_colors_fetcher(x), axis=1)

    df_links.loc[:, 'value'] = df_links[col_counts]

    return df_nodes, df_links
    



