import numpy as np


def load_abide(path):
    """Load the ABIDE-I CC200 npz. Returns connectome edge vectors and phenotypes."""
    d = np.load(path, allow_pickle=True)
    return dict(
        X=np.nan_to_num(d["connectomes"].astype(np.float32), nan=0.0),
        site=np.asarray(d["site"], dtype=str),
        age=d["age"].astype(np.float32),
        fiq=d["fiq"].astype(np.float32),
        asd=d["asd"].astype(np.float32),
    )


def target_config(data):
    return {
        "age": dict(y=data["age"], valid=~np.isnan(data["age"]), task="reg"),
        "fiq": dict(y=data["fiq"], valid=~np.isnan(data["fiq"]), task="reg"),
        "asd": dict(y=data["asd"], valid=~np.isnan(data["asd"]), task="cls"),
    }


def site_indices(valid, site, site_name):
    """Global indices of subjects at `site_name` with a valid label."""
    return np.where(valid & (site == site_name))[0]


def train_test_split(n_total, n_train, seed):
    """Random split inside the site: first n_train train, the rest test."""
    idx = np.random.RandomState(seed).permutation(n_total)
    return idx[:n_train], idx[n_train:]
