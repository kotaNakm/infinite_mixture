import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import pandas as pd
import sys
from importlib import import_module
from tqdm import tqdm, trange
from sklearn.feature_extraction.text import CountVectorizer


def prepare_event_tensor(
    given_data,
    categorical_idxs,
    time_idx,
    freq,
    edited_time=False,
    return_encoders=False,
    save_encoders=False,
    outdir="./",
):
    data = given_data.copy("deep")
    print(data)
    print(data.columns)
    data = data.dropna(subset=(categorical_idxs + [time_idx]))

    # Add date information
    data["weekday"] = data[time_idx].dt.day_name()
    data["month"] = data[time_idx].dt.month_name()
    data["hour"] = data[time_idx].dt.hour
    data["weeklyhour"] = data["weekday"] + data["hour"].map(str)
    tmp_idxs = categorical_idxs + ["weekday", "month", "hour", "weeklyhour"]

    # Encode timestamps
    data[time_idx] = data[time_idx].round(freq)
    data = data.sort_values(time_idx)
    start = data[time_idx].min()
    end = data[time_idx].max()
    ticks = pd.date_range(start, end, freq=freq)
    timepoint_encoder = preprocessing.LabelEncoder()
    timepoint_encoder.fit(ticks)
    data[time_idx] = timepoint_encoder.transform(data[time_idx].values)

    # Encode categorical data
    oe = preprocessing.OrdinalEncoder()
    data[tmp_idxs] = oe.fit_transform(data[tmp_idxs])
    data[tmp_idxs] = data[tmp_idxs].astype(int)

    if save_encoders:
        # Timestamps
        time_encoder = pd.DataFrame(
            timepoint_encoder.classes_,
            index=range(len(timepoint_encoder.classes_)),
            columns=["timestamp"],
        )

        time_encoder.to_csv(outdir + f"/{time_idx}.csv.gz", index=False)

        # Categorical features
        for key, feature_elem in zip(tmp_idxs, oe.categories_):
            ctg_encoder = pd.DataFrame(
                feature_elem, index=range(len(feature_elem)), columns=[key]
            )
            ctg_encoder.to_csv(outdir + "/" + key + ".csv.gz", index=False)

    if return_encoders:
        return data.reset_index(drop=True), oe, timepoint_encoder
    else:
        return data.reset_index(drop=True)


def prepare_event_matrix(
    given_data,
    categorical_idxs,
    return_encoders=False,
    save_encoders=False,
    outdir="./",
):
    data = given_data.copy("deep")
    data = data.dropna(subset=categorical_idxs)
    print(data)
    print(data.columns)

    # Encode categorical data
    oe = preprocessing.OrdinalEncoder()
    data[categorical_idxs] = oe.fit_transform(data[categorical_idxs])
    data[categorical_idxs] = data[categorical_idxs].astype(int)

    if save_encoders:
        # Categorical features
        for key, feature_elem in zip(categorical_idxs, oe.categories_):
            ctg_encoder = pd.DataFrame(
                feature_elem, index=range(len(feature_elem)), columns=[key]
            )
            ctg_encoder.to_csv(outdir + "/" + key + ".csv.gz", index=False)

    if return_encoders:
        return data.reset_index(drop=True), oe
    else:
        return data.reset_index(drop=True)


def transform_event_tensor_to_document(
    event_tensor,
    tensor_shape,
    outdir,
    time_ind=0,
    time_base=False,
):
    """
    first column have to be time index
    only vectorize exisiting combinations of units/dimensions in each of attributes
    """
    keys = event_tensor.columns
    with open(outdir + "/keys.csv", "w") as f:
        f.write(",".join(keys))

    # make tensor and then unfold
    # 1. make tensor
    tensor_numpy = np.zeros(tensor_shape)
    for key_, df in event_tensor.groupby(list(keys)):
        # tensor_numpy[key_] = df.size
        tensor_numpy[key_] = len(df)

    if time_base:
        X = tensor_numpy.reshape(tensor_numpy.shape[0], -1)
    else:
        X = tensor_numpy.reshape(1, -1)

    return X


def transform_event_tensor_to_document_count_vector(
    given_event_tensor,
    time_ind=0,
    outdir="./",
):
    """
    first column have to be time index
    genenrate (time x attributes) dataset as (documents x words)

    genenrate (time ind attribute x other attributes) dataset as (documents x words)
    """

    event_tensor = given_event_tensor.copy("deep")
    keys = event_tensor.columns
    with open(outdir + "/keys.csv", "w") as f:
        f.write(",".join(keys))

    base_key = event_tensor.columns[0]
    feature_keys = event_tensor.columns[1:]

    # transform feature_keys to like words
    f_mode = len(feature_keys)
    for key_id, f_key in enumerate(feature_keys):
        temp_values = event_tensor[f_key].values
        n_digit = len(str(temp_values.max()))
        temp_values = [
            "entity" + str(key_id).zfill(f_mode) + "_" + str(i).zfill(n_digit)
            for i in temp_values
        ]
        event_tensor[f_key] = temp_values

    corpus = {}
    vectorizer = CountVectorizer()
    for key_i, g in tqdm(event_tensor.groupby(base_key), desc="bow"):
        # make corpus with all included words in key_i rows
        corpus[key_i] = " ".join(g[feature_keys].values.ravel().tolist())
    # generate vectors based on sorted values
    count_vectors = vectorizer.fit_transform(corpus.values())
    count_vectors = count_vectors.toarray()

    # refine count vectors because time indexs has sparse
    n_time_idxs = event_tensor[base_key].max() + 1
    extend_count_vectors = np.zeros((n_time_idxs, count_vectors.shape[1]))

    time_idxs = np.array(event_tensor[base_key].unique(), dtype=int)
    print(count_vectors)
    print(time_idxs)
    extend_count_vectors[time_idxs] = count_vectors

    # bow_dict = vectorizer.vocabulary_
    # bow_dict = {v: k for k, v in vectorizer.vocabulary_.items()}

    # # check order of vectors
    # print(count_vectors.shape)
    # print(vectorizer.vocabulary_)
    # sorted_vocab = dict(sorted(vectorizer.vocabulary_.items(), key=lambda x:x[0]))
    # print(sorted_vocab)
    # print(sorted_vocab.values())
    # print(type(sorted_vocab))
    # print(count_vectors.shape)

    return extend_count_vectors, event_tensor, vectorizer


def compute_bow(
    data, base_key, time_key, feature_keys, freq="D", calibration_period_end=None
):

    vectorizer = CountVectorizer()
    corpus = {}

    # Extract transactions during the calibration period
    if calibration_period_end is not None:
        calib_data = data[lambda x: x[time_key] <= calibration_period_end].copy()
    else:
        calib_data = data.copy()

    if freq == "D":
        time_format = "%Y_%m_%d"
    elif freq == "H":
        time_format = "%Y_%m_%d_%H"

    calib_data[time_key] = calib_data[time_key].round(freq)
    calib_data[time_key] = calib_data[time_key].dt.strftime(time_format)

    # preprocessing
    calib_data[feature_keys] = calib_data[feature_keys].astype(str)
    for key in tqdm(feature_keys, desc="preprocessing"):
        calib_data[key] = calib_data[key].str.replace(" ", "")
        calib_data[key] = calib_data[key].str.replace("-", "_")

    # text to count vectors
    # based on last key
    # ravel is equivalent to reshape(-1, order=order)
    for key_i, g in tqdm(calib_data.groupby(base_key), desc="bow"):
        # make corpus with all included words in key_i rows
        corpus[key_i] = " ".join(g[feature_keys].values.ravel().tolist())
        # print(g[feature_keys].values.ravel().tolist())
        # print(corpus[key_i])
        # exit()
    count_vectors = vectorizer.fit_transform(corpus.values())
    count_vectors = count_vectors.toarray()

    # bow_dict = vectorizer.vocabulary_
    bow_dict = {v: k for k, v in vectorizer.vocabulary_.items()}

    return count_vectors, corpus, bow_dict


def import_dataframe(args, synthetic_tag=""):
    input_tag = args.input_tag
    import_type = args.import_type
    sys.path.append("_dat")

    dataset_module = import_module(input_tag)

    # For import of synthetic data
    if len(synthetic_tag) > 0:
        return dataset_module.load_clean_data(synthetic_tag)

    if import_type == "clean_data":
        return dataset_module.load_clean_data()
    elif import_type == "clean_quater":
        return dataset_module.load_clean_quater()
    elif import_type == "clean_month":
        return dataset_module.load_clean_a_month()
    elif import_type == "clean_week":
        return dataset_module.load_clean_a_week()
    elif import_type == "clean_frequent":
        return dataset_module.load_clean_frequent()
    elif import_type == "clean_201920":
        return dataset_module.load_clean_201920()
    elif import_type == "with_price_rank":
        return dataset_module.load_data_with_price_rank()
    elif import_type == "downsample_20":
        return dataset_module.load_data_attack20()
    elif import_type == "downsample_20_short":
        return dataset_module.load_data_attack20(short=True)
    elif import_type == "dos1":
        return dataset_module.load_data(type="dos1")
    else:
        exit("Not supported import type")
