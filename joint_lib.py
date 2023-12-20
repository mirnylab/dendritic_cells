import numpy as np
import pandas as pd

from cooltools.lib import numutils
from cooltools.api.eigdecomp import _filter_heatmap, _fake_cis

import inspectro
import inspectro.utils
from inspectro.utils.eigdecomp import _normalized_affinity_matrix_from_cis

import bioframe
import cooler

import pySTATIS

import tqdm

from cooltools.lib import numutils

### Plotting utils ###
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
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


def _affinity_matrix_cis(A, perc_top, perc_bottom, ignore_diags):
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
    numutils.set_diag(A, 0, 0)
    OE = numutils.iterative_correction_symmetric(OE)[0]
    marg = np.r_[np.sum(OE, axis=0), np.sum(OE, axis=1)]
    marg = np.mean(marg[marg > 0])
    OE /= marg

    # empty invalid rows, so that get_eig can find them
    OE[is_bad_bin, :] = 0
    OE[:, is_bad_bin] = 0

    return OE


def _affinity_matrix_trans(A, partition, perc_top, perc_bottom):
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
    for n, (i0, i1) in enumerate(extents):
        A[i0:i1, i0:i1] = 0
        part_ids.extend([n] * (i1 - i0))
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


def randomize_symm(mtx, replace=False):
    """ Randomize a symmetric matrix, preserving the marginal sums of rows.
    The randoization is performed by shuffling the upper triangle of the matrix by row
    (sampling from the row values without replacement by default).
    The matrix is the symmetrizes by transposing. 
    Parameters:
    -----------
    mtx : 2D array
        Symmetric matrix to randomize.
    replace : bool, optional
        Whether to sample with replacement (default False).
    """
    mtx_output = np.zeros(mtx.shape)
    l = mtx_output.shape[0]
    
    for i in range(l):
        mtx_output[i, i+1:] = mtx[i, i+1+np.random.choice(l-i-1, size=l-i-1, replace=replace)]
    
    # Symmetrize:
    mtx_output += mtx_output.T
    
    return mtx_output

def _affinity_matrix_cis_randomized(A, perc_top, perc_bottom, ignore_diags):
    """
    Randomize a symmetric matrix, preserving the marginal sums of rows.
    Take Hi-C matrix, normalize by OOE, randomize, zero out diagonal elements and then perform ICE.

    Parameters:
    -----------
    A : 2D array
        Symmetric matrix to randomize.
    perc_top : float
        Percentile of top values to keep.
    perc_bottom : float 
        Percentile of bottom values to keep.
    ignore_diags : int
        Number of diagonals to ignore.
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
    OE = randomize_symm(OE, replace=False)
    
    # Inject zero diagonal, balance and rescale margins to 1
    numutils.set_diag(A, 0, 0)
    
    OE = numutils.iterative_correction_symmetric(OE)[0]
    marg = np.r_[np.sum(OE, axis=0), np.sum(OE, axis=1)]
    marg = np.mean(marg[marg > 0])
    OE /= marg

    # empty invalid rows, so that get_eig can find them
    OE[is_bad_bin, :] = 0
    OE[:, is_bad_bin] = 0

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
    func_retrieve=_affinity_matrix_cis,
    nthreads=1,  # currently not tested
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
        List of bad bins.
        """
    if nthreads > 1:
        warnings.warn("Nthreads > 1 currently not implemented")
    nthreads = 1

    # get chromosomes from cooler, if view_df not specified:
    if view_df is None:
        view_df = bioframe.make_viewframe(
            [(chrom, 0, clr.chromsizes[chrom]) for chrom in clr.chromnames]
        )
    else:
        # appropriate viewframe checks:
        if not bioframe.is_viewframe(view_df):
            raise ValueError("view_df is not a valid viewframe.")
        if not bioframe.is_contained(view_df, bioframe.make_viewframe(clr.chromsizes)):
            raise ValueError("view_df is out of the bounds of chromosomes in cooler.")

    if 'name' not in view_df.columns:
        view_df = bioframe.make_viewframe(view_df, name_style="ucsc")

    # ignore diags as in cooler inless specified
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

        A = func_retrieve(A, perc_top, perc_bottom, ignore_diags)

        return _region, A

    # return matrix per region (can be multiprocessed)
    if verbose:
        if nthreads > 1:
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
            with Pool(2) as p:
                results = list(p.map(_each, view_df.values))
        else:
            results = list(map(_each, view_df.values))

    # results = map(_each, view_df.values)

    output = {}
    for _region, A in results:
        output[f"{_region[3]}"] = A.copy()

    return output


def retrieve_trans_matrices(
    clr,
    # view_df=None, # Currently not used
    # ignore_diags=None,
    partition=None,
    # bad_bins=None,
    balance="weight",
    perc_bottom=1,
    perc_top=99.95,
):
    if partition is None:
        partition = np.r_[
            [clr.offset(chrom) for chrom in clr.chromnames], len(clr.bins())
        ]
    lo = partition[0]
    hi = partition[-1]

    A = clr.matrix(balance=balance)[lo:hi, lo:hi]

    A = _affinity_matrix_trans(A, partition, perc_top, perc_bottom)

    return A.copy()


def preload_coolers(COOLER_PATHS, CONDITIONS, BINSIZE):
    dict_coolers = {}
    for cond in CONDITIONS:
        path = COOLER_PATHS[cond]

        clr = cooler.Cooler(path + f"::resolutions/{BINSIZE}")

        dict_coolers[cond] = clr

    return dict_coolers


def load_cis_matrices(COOLER_DICT, CONDITIONS, viewframe=None, ignore_diags=None):
    OUTPUT = {}
    for cond in (pbar := tqdm.tqdm(CONDITIONS)):
        pbar.set_postfix_str(cond)

        clr = COOLER_DICT[cond]

        if viewframe is None:
            viewframe = bioframe.make_viewframe(clr.chromsizes)

        OUTPUT[cond] = dict(retrieve_cis_matrices(
            clr=clr,
            view_df=viewframe,
            verbose=True,
            ignore_diags=ignore_diags,
            # nthreads=1
        ))

    return OUTPUT


def load_trans_matrices(COOLER_DICT, CONDITIONS, CHROMOSOMES=None):
    OUTPUT = {}
    for cond in (pbar := tqdm.tqdm(CONDITIONS)):
        pbar.set_postfix_str(cond)

        clr = COOLER_DICT[cond]

        if CHROMOSOMES is None:
            CHROMOSOMES = clr.chromnames

        partition = np.r_[
            [clr.offset(chrom) for chrom in CHROMOSOMES], clr.extent(CHROMOSOMES[-1])[1]
        ]

        OUTPUT[cond] = retrieve_trans_matrices(
            clr=clr,
            partition=partition,
        )

    return OUTPUT


def list_to_filtered_array(
    COOLERS_DICT,
    bad_bins=None,
    good_bins=None,
    conditions=None,
    backend="STATIS",
    norm=["norm_one"],
):
    """
    norm: normalization method to use (None, 'zscore', 'double_center')
    """
    if conditions is None:
        conditions = list(COOLERS_DICT.keys())

    datasets = np.array([COOLERS_DICT[cond] for cond in conditions])

    if good_bins is None:
        good_bins = np.all(datasets.sum(axis=1) != 0, axis=0)

    if not bad_bins is None:
        good_bins[bad_bins] = False

    # Print stats of bad bins removal
    for i, cond in enumerate(conditions):
        matrix = COOLERS_DICT[cond][good_bins, :][:, good_bins]  # remove the empty bins
        prc_zeros_init = (
            100
            * np.sum(datasets[i, :, :] == 0)
            / np.float(datasets[i, :, :].shape[0] * datasets[i, :, :].shape[1])
        )
        prc_zeros_result = (
            100 * np.sum(matrix == 0) / np.float(matrix.shape[0] * matrix.shape[1])
        )
        print(
            f"""{cond}
        zeros ratio before zero rows removal: {prc_zeros_init:.2f}% 
        zeros ratio after zero rows removal: {prc_zeros_result:.2f}%
        """
        )

    X = []
    for i, cond in enumerate(conditions):
        matrix = COOLERS_DICT[cond][good_bins, :][:, good_bins]  # remove the empty bins

        if backend == "STATIS":
            data = pySTATIS.STATISData(matrix, cond, normalize=norm)
        else:
            data = matrix.copy()

        X.append(data)

    return X, good_bins


def run_pySTATIS(X, n_comp):
    st = pySTATIS.STATIS(n_comps=n_comp)
    st.fit(X)

    return st


def summary_STATIS(st, CONDITIONS, n_comp_plot=10, COMP_TARGET=1, REGION=(0, 1000)):
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

    n_comp = st.D_.shape[0]
    NBINS = st.factor_scores_.shape[0]

    print("Eigenvalues (D) plot")
    plt.figure(figsize=[10, 5])
    sns.barplot(y=st.D_, x=np.arange(n_comp), color="grey")  # list of eigenvalues
    plt.xticks(np.arange(0, n_comp, 5), rotation=90)
    plt.title("Eigenvalues")
    plt.show()

    print("Percentage of variance explained:")
    print((100 * np.power(st.D_, 2) / sum(np.power(st.D_, 2)))[:n_comp_plot])

    # st.print_variance_explained()

    print(
        "Weights of each dataset, assessing the contribution of each dataset to the compromise"
    )
    print(st.table_weights_)

    print("Barplot of weights")
    plt.figure(figsize=[10, 5])
    sns.barplot(y=st.table_weights_, x=CONDITIONS, color="grey")
    plt.xticks(rotation=90)
    plt.title("Weights of the datasets, a")
    plt.tight_layout()
    plt.show()

    # What datasets determined the components?
    print("Heatmap of contributions of datasets to components")
    plt.figure(figsize=[20, 5])
    sns.heatmap(st.contrib_dat_[:, :n_comp_plot], square=True, cmap="Reds")
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
        plt.plot(st.contrib_obs_[REGION[0] : REGION[1], i], label=f"PC{i}")
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
            st.contrib_var_[k * NBINS : (k + 1) * NBINS, i], label=f"PC{i} for {cond}"
        )
    plt.xlim(0, 1_000)  # NBINS)
    plt.legend()
    plt.title("Contribution of genomic positions (observations) to the components:")
    plt.tight_layout()

    print("Quick scatterplots of loadings")
    Q = st.Q_
    i1 = COMP_TARGET  # PC1
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
        points = sns.scatterplot(
            x=Q[k * NBINS : (k + 1) * NBINS, i1],
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
    F = st.factor_scores_
    Fp = st.partial_factor_scores_

    i1 = COMP_TARGET  # PC1
    i2 = COMP_TARGET + 1  # PC2

    fig, axes = plt.subplots(
        n_rows + 1, n_cols, squeeze=False, figsize=[3 * n_cols, 3 * (n_rows + 1)]
    )
    axes = axes.flatten()

    for k, cond in enumerate(CONDITIONS):
        ax = axes[k]
        points = sns.scatterplot(
            x=Fp[k, :, i1],
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
        x=F[:, i1],
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


def parse_statis_output(st, bins, conditions, good_bins=None, n_comp=10, postfix=""):
    """ Add columns to the bin table corresponding to GSVD output (STATIS-type):
    - loadings
    - loadings normalized by square of singular values
    - compromise (consensus dimensions)
    - factor scores (partial factor scores for each dataset)

    Optionally, add postfix to each parsed column output.
    """
    bins = bins.copy()

    if n_comp is None:
        n_comp = st.D_.shape[0]
    else:
        n_comp = n_comp
    NBINS = st.factor_scores_.shape[0]

    bins.loc[:, "is_bin_valid"] = False
    bins.loc[good_bins, "is_bin_valid"] = True

    assert NBINS == len(bins.query("is_bin_valid==True"))

    # Loadings:
    for k, cond in enumerate(conditions):
        for i in range(n_comp):
            loading = st.Q_[k * NBINS : (k + 1) * NBINS, i]
            bins.loc[good_bins, f"loading:{cond}:{i}"+postfix] = loading

    # Loadings normalized by square of singular values:
    for k, cond in enumerate(conditions):
        for i in range(n_comp):
            loading = st.Q_[k * NBINS : (k + 1) * NBINS, i]
            sv = st.D_[i]
            bins.loc[good_bins, f"loading-norm:{cond}:{i}"+postfix] = loading * sv**2

    # Compromise:
    for i in range(n_comp):
        dimension = st.factor_scores_[:, i]
        bins.loc[good_bins, f"consensus-dimension:{i}"+postfix] = dimension

    # Factor scores:
    for k, cond in enumerate(conditions):
        for i in range(n_comp):
            dimension = st.partial_factor_scores_[k, :, i]
            bins.loc[good_bins, f"factor-scores:{cond}:{i}"+postfix] = dimension

    variance_explained = np.power(st.D_, 2) / sum(np.power(st.D_, 2))
    return bins, variance_explained


def projection(A, reference, n_comp=None, normalize=True):
    """
    Inspectro function to project matrix A to the eigenvectors eigs:
    https://github.com/open2c/inspectro/blob/94c3e1897f6a709b843f3604bb97dfa32e6cd3db/inspectro/utils/eigdecomp.py#L541

    A : Input matrix
    reference : eigenvector or factor scores DataFrame or matrix
    n_comp : limiting number of eigenvectors/factor scores

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


def transform(X, reference, variance_explained=None, good_bins=None, n_comp=None, whiten=True):
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

def groups2sankey_multicolor(df, states, n_states, n_clusters, cmap):
    """ More advanced implementation of df_links that sorts the links by nodes appearance"""
    clusters_numbered = np.arange(n_clusters)
    colors_clusters = [ mpl.colors.rgb2hex(x) for x in cmap.colors[:n_clusters]]
    colors_clusters_alpha = [ "rgba("+",".join( [str(x) for x in ((*mpl.colors.hex2color( mpl.colors.rgb2hex(x) ), 0.2))] )+")" for x in cmap.colors[:n_clusters]]
    
    df_nodes = pd.DataFrame({
        'cluster': list(clusters_numbered) * n_states, 
        'color': list(colors_clusters) * n_states,
        'cond': np.repeat(conditions, n_clusters),
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


def groups2sankey_binary_color(df, cols_states, n_states, n_clusters, cmap):
    """ More advanced implementation of df_links that sorts the links by nodes appearance"""
    clusters_numbered = np.arange(n_clusters)
    colors_clusters = [ mpl.colors.rgb2hex(x) for x in cmap.colors[:n_clusters]]
    colors_clusters_alpha = [ "rgba("+",".join( [str(x) for x in ((*mpl.colors.hex2color( mpl.colors.rgb2hex(x) ), 0.2))] )+")" for x in cmap.colors[:n_clusters]]
    
    df_nodes = pd.DataFrame({
        'cluster': list(clusters_numbered) * n_states, 
        'color': list(colors_clusters) * n_states,
        'cond': np.repeat(conditions, n_clusters),
        'x' : np.repeat(range(n_states), n_clusters)*10
    })
    
    df_links = df.copy()

    for i in range(n_states):
        df_links.loc[:, "true" + cols_states[i]] = df_links.loc[:, cols_states[i]].astype(int)
        df_links.loc[:, f"color {cols_states[i]}"] = [colors_clusters_alpha[i] for i in df_links.loc[:, cols_states[i]].astype(int).values ]
        df_links.loc[:, cols_states[i]] = df_links.loc[:, cols_states[i]].astype(int) + i*(n_clusters)

#     df_links.loc[:, 'color_left']  = df_links.apply(lambda x: x[f"color {cols_states[0]}"] if x["true" + cols_states[1]]!=x["true" + cols_states[2]] else add_alpha('gray', 0.2), axis=1)
#     df_links.loc[:, 'color_right'] = df_links.apply(lambda x: x[f"color {cols_states[2]}"] if x["true" + cols_states[1]]!=x["true" + cols_states[2]] else add_alpha('gray', 0.2), axis=1)

    df_links.loc[:, 'color_left']  = df_links.apply(lambda x: add_alpha('red', 0.2) if x["true" + cols_states[1]]!=x["true" + cols_states[-1]] else add_alpha('gray', 0.2), axis=1)
    df_links.loc[:, 'color_right'] = df_links.apply(lambda x: add_alpha('red', 0.2) if x["true" + cols_states[1]]!=x["true" + cols_states[-1]] else add_alpha('gray', 0.2), axis=1)

    dfs = []
    for state1, state2 in zip(cols_states[:-1], cols_states[1:]):
        df_tmp = df_links.loc[:, [state1, state2, 'nbins', 'color_left']+list( df_links.columns )]
        df_tmp.columns=['left', 'right', 'value', 'color']+list( df_links.columns )
        
        dfs.append(df_tmp.copy())

    df_links = pd.concat(dfs, axis=0).sort_values(["true" + state for state in cols_states])
    
    return df_nodes, df_links