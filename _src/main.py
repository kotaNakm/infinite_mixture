import argparse
import os
import shutil
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from importlib import import_module
import sys
from seaborn import widgets
import dill


sys.path.append("_dat")
sys.path.append("_src")

import utils
from infinite_mixture_unigram import infinite_mixture_unigram

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input/Output
    parser.add_argument("--input_tag", type=str)  #
    parser.add_argument("--out_dir", type=str)  #
    parser.add_argument("--categorical_idxs", type=str)
    parser.add_argument("--import_type", type=str, default="clean_data")

    # model
    parser.add_argument("--N_ITER", type=int, default=20)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--beta", type=float)

    # experiments
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # make output dir
    outputdir = args.out_dir
    if os.path.exists(outputdir):
        shutil.rmtree(outputdir)
    os.makedirs(outputdir)

    # data input
    dataset_module = import_module(args.input_tag)
    categorical_idxs = args.categorical_idxs.split("/")

    raw_df = utils.import_dataframe(args)

    # for debag short version
    # raw_df = raw_df.iloc[:1000]

    matrix_df = utils.prepare_event_matrix(
        raw_df,
        categorical_idxs,
        outdir=outputdir,
        save_encoders=True,
    )

    matrix_df = matrix_df[categorical_idxs]

    mat_shape = matrix_df.max().values + 1
    n_full_cells = len(matrix_df.groupby(["brand", "category_code"]).count())
    # n_full_cells = np.sum(matrix_df.groupby(categorical_idxs).size().values)

    print(f"--Dataset description--")
    print(f"matrix shape: {mat_shape}")
    print(f"# of records: {len(matrix_df)}")
    print(f"sparsity(%): {1 - n_full_cells/ np.prod(mat_shape)}")
    print(f"------------------------")

    start_time = time.process_time()
    inf_unigram = infinite_mixture_unigram(
        alpha=args.alpha,
        beta=args.beta,
        max_iter=args.N_ITER,
        random_state=0,
        verbose=args.verbose,
    )

    W = inf_unigram.fit_transform(matrix_df)
    elapsed_time = time.process_time() - start_time
    print("done infinite mixture unigram (CRP) inference")

    result = [inf_unigram, elapsed_time]
    if True:
        dill.dump(result, open(f"{outputdir}/result.dill", "wb"))

    print(f"elapsed_time:{elapsed_time}s")
    print(outputdir)
