import pandas as pd


def category2int(col):
    if col.dtype == object:
        mapping = {label: idx for idx, label in enumerate(set(col))}
        return col.map(mapping)
    else:
        return col


if __name__ == "__main__":
    df = pd.read_csv("application_train.csv")
    df = df.apply(category2int, axis=0)
    df.to_hdf("../clean_data/main_train.h5", key="SK_ID_CURR")
