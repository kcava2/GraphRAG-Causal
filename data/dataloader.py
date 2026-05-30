import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE


def load_and_clean(filepath):
    df = pd.read_csv(filepath)
    df = df.rename(columns={
        "merged_air_crash_data_weather.sky_cond_nonceil": "SkyCondNonceil"
    })
    # Fill NaN in categorical inputs before splitting so encoders see all categories
    df["WeatherCondition"]    = df["WeatherCondition"].fillna("Unknown")
    df["SkyCondNonceil"]      = df["SkyCondNonceil"].fillna("Unknown")
    df["Personnel Conditions"] = df["Personnel Conditions"].fillna("Unknown")
    # Drop the 6 rows where Supervisory Conditions (step-0 target) is NaN
    df = df.dropna(subset=["Supervisory Conditions", "Operator Conditions", "Unsafe Conditions"])
    return df.reset_index(drop=True)


class SCAMAAPEncoders:
    """
    Fits all label encoders on the training split.

    Step A inputs : Organizational Climate, Employment (numerical)
    Step 0 inputs : WeatherCondition, TimeOfDay, SkyCondNonceil,
                    Personnel Conditions, Supervisory Conditions
    Step 1 inputs : soft_B, Supervisory Conditions
    Targets       : Supervisory Conditions (A), Operator Conditions (B), Unsafe Conditions (C)
    """

    def __init__(self, df):
        self.enc_org_climate  = LabelEncoder().fit(df["Organizational Climate"])
        self.enc_weather      = LabelEncoder().fit(df["WeatherCondition"])
        self.enc_time_of_day  = LabelEncoder().fit(df["TimeOfDay"])
        self.enc_sky_cond     = LabelEncoder().fit(df["SkyCondNonceil"])
        self.enc_personnel    = LabelEncoder().fit(df["Personnel Conditions"])
        self.enc_supervisory  = LabelEncoder().fit(df["Supervisory Conditions"])

        self.enc_operator     = LabelEncoder().fit(df["Operator Conditions"])
        self.enc_unsafe       = LabelEncoder().fit(df["Unsafe Conditions"])


class SCAMAAPSequenceDataset(Dataset):
    """
    Each sample feeds three causal LSTM steps.

    Step A: [Organizational Climate | Employment]  dim=2  →  predict A: Supervisory Conditions
    Step 0: [WeatherCondition | TimeOfDay | SkyCondNonceil | Personnel | Supervisory]
            dim=5  →  predict B: Operator Conditions
    Step 1 (fully dynamic – computed in model forward):
            [soft_B | Supervisory]  →  predict C: Unsafe Acts
    """

    def __init__(self, df, encoders: SCAMAAPEncoders):
        e = encoders
        df = df.copy()

        # ── Step-A categorical + numerical inputs ────────────────────────────
        org_climate  = e.enc_org_climate.transform(df["Organizational Climate"]).astype("float32")
        employment   = df["Employment, Total Weighted Avg CY_QoQ_pct"].values.astype("float32")

        self.step_a = torch.tensor(
            list(zip(org_climate, employment)),
            dtype=torch.float32,
        )  # (N, 2)

        # ── Step-0 categorical inputs ────────────────────────────────────────
        weather      = e.enc_weather.transform(df["WeatherCondition"]).astype("float32")
        time_of_day  = e.enc_time_of_day.transform(df["TimeOfDay"]).astype("float32")
        sky_cond     = e.enc_sky_cond.transform(df["SkyCondNonceil"]).astype("float32")
        personnel    = e.enc_personnel.transform(df["Personnel Conditions"]).astype("float32")
        supervisory  = e.enc_supervisory.transform(df["Supervisory Conditions"]).astype("float32")

        self.step0 = torch.tensor(
            list(zip(weather, time_of_day, sky_cond, personnel, supervisory)),
            dtype=torch.float32,
        )  # (N, 5)

        # ── Targets ─────────────────────────────────────────────────────────
        self.y_A = torch.tensor(
            e.enc_supervisory.transform(df["Supervisory Conditions"]), dtype=torch.long
        )
        self.y_B = torch.tensor(
            e.enc_operator.transform(df["Operator Conditions"]), dtype=torch.long
        )
        self.y_C = torch.tensor(
            e.enc_unsafe.transform(df["Unsafe Conditions"]), dtype=torch.long
        )

    def __len__(self):
        return len(self.y_B)

    def __getitem__(self, idx):
        return (
            self.step_a[idx],  # (2,)
            self.step0[idx],   # (5,)
            self.y_A[idx],
            self.y_B[idx],
            self.y_C[idx],
        )


class _SMOTEDataset(Dataset):
    """Wraps SMOTE-resampled numpy arrays as a Dataset."""
    def __init__(self, step_a, X, y_A, y_B, y_C):
        self.step_a = torch.tensor(step_a, dtype=torch.float32)
        self.step0  = torch.tensor(X, dtype=torch.float32)
        self.y_A    = torch.tensor(y_A, dtype=torch.long)
        self.y_B    = torch.tensor(y_B, dtype=torch.long)
        self.y_C    = torch.tensor(y_C, dtype=torch.long)

    def __len__(self):
        return len(self.y_B)

    def __getitem__(self, idx):
        return self.step_a[idx], self.step0[idx], self.y_A[idx], self.y_B[idx], self.y_C[idx]


def get_dataloaders(filepath, test_split=0.2, val_split=0.1, batch_size=32, seed=42):
    """
    Returns (train_loader, val_loader, test_loader, encoders).

    Encoders are fit exclusively on the training split so there is no
    data leakage into validation or test.
    """
    df = load_and_clean(filepath)
    n = len(df)

    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=rng).tolist()

    n_test  = int(n * test_split)
    n_val   = int(n * val_split)
    n_train = n - n_test - n_val

    df_train = df.iloc[perm[:n_train]].reset_index(drop=True)
    df_val   = df.iloc[perm[n_train : n_train + n_val]].reset_index(drop=True)
    df_test  = df.iloc[perm[n_train + n_val :]].reset_index(drop=True)

    encoders = SCAMAAPEncoders(df_train)

    # SMOTE on training split only — oversample minority classes for all three targets (A, B, C)
    _enc = encoders

    def _step0_array(df):
        return np.column_stack([
            _enc.enc_weather.transform(df["WeatherCondition"]),
            _enc.enc_time_of_day.transform(df["TimeOfDay"]),
            _enc.enc_sky_cond.transform(df["SkyCondNonceil"]),
            _enc.enc_personnel.transform(df["Personnel Conditions"]),
            _enc.enc_supervisory.transform(df["Supervisory Conditions"]),
        ])

    def _step_a_array(df):
        return np.column_stack([
            _enc.enc_org_climate.transform(df["Organizational Climate"]),
            df["Employment, Total Weighted Avg CY_QoQ_pct"].values,
        ])

    X_train       = _step0_array(df_train)
    step_a_train  = _step_a_array(df_train)
    y_A_train     = _enc.enc_supervisory.transform(df_train["Supervisory Conditions"])
    y_B_train     = _enc.enc_operator.transform(df_train["Operator Conditions"])
    y_C_train     = _enc.enc_unsafe.transform(df_train["Unsafe Conditions"])

    from sklearn.neighbors import NearestNeighbors

    def _safe_smote(X, y, seed, target_ratio=0.3):
        """
        Run partial SMOTE — brings minority classes to `target_ratio * majority_count`
        instead of full balance. Reduces synthetic inflation while still addressing
        the worst class imbalance.
        Returns (X_synthetic, y_synthetic) — only the NEW synthetic rows, not the originals.
        Returns (None, None) if oversampling is not possible or not needed.
        """
        counts = np.bincount(y)
        min_count = counts[counts > 0].min()
        if min_count < 2:
            return None, None
        majority_count = counts.max()
        target_count = int(majority_count * target_ratio)
        strategy = {
            cls: target_count
            for cls, cnt in enumerate(counts)
            if 0 < cnt < target_count
        }
        if not strategy:
            return None, None
        k = min(5, min_count - 1)
        sm = SMOTETomek(
            smote=SMOTE(random_state=seed, k_neighbors=k, sampling_strategy=strategy),
            random_state=seed,
        )
        X_res, y_res = sm.fit_resample(X, y)
        return X_res[len(X):], y_res[len(y):]

    # Single NN lookup — fit once on original X_train feature space
    nn_lookup = NearestNeighbors(n_neighbors=1).fit(X_train)

    def _inherit(X_synth, arrays):
        """Find nearest original row for each synthetic row; return inherited label arrays."""
        _, indices = nn_lookup.kneighbors(X_synth)
        idx = indices[:, 0]
        return [arr[idx] for arr in arrays]

    # ── SMOTE for A ──────────────────────────────────────────────────────────
    X_synth_A, y_A_synth = _safe_smote(X_train, y_A_train, seed)
    if X_synth_A is not None:
        step_a_synth_A, y_B_from_A, y_C_from_A = _inherit(
            X_synth_A, [step_a_train, y_B_train, y_C_train]
        )
    else:
        X_synth_A      = np.empty((0, X_train.shape[1]))
        y_A_synth      = np.empty(0, dtype=y_A_train.dtype)
        step_a_synth_A = np.empty((0, step_a_train.shape[1]))
        y_B_from_A     = np.empty(0, dtype=y_B_train.dtype)
        y_C_from_A     = np.empty(0, dtype=y_C_train.dtype)

    # ── SMOTE for B ──────────────────────────────────────────────────────────
    X_synth_B, y_B_synth = _safe_smote(X_train, y_B_train, seed)
    if X_synth_B is not None:
        step_a_synth_B, y_A_from_B, y_C_from_B = _inherit(
            X_synth_B, [step_a_train, y_A_train, y_C_train]
        )
    else:
        X_synth_B      = np.empty((0, X_train.shape[1]))
        y_B_synth      = np.empty(0, dtype=y_B_train.dtype)
        step_a_synth_B = np.empty((0, step_a_train.shape[1]))
        y_A_from_B     = np.empty(0, dtype=y_A_train.dtype)
        y_C_from_B     = np.empty(0, dtype=y_C_train.dtype)

    # ── SMOTE for C ──────────────────────────────────────────────────────────
    X_synth_C, y_C_synth = _safe_smote(X_train, y_C_train, seed)
    if X_synth_C is not None:
        step_a_synth_C, y_A_from_C, y_B_from_C = _inherit(
            X_synth_C, [step_a_train, y_A_train, y_B_train]
        )
    else:
        X_synth_C      = np.empty((0, X_train.shape[1]))
        y_C_synth      = np.empty(0, dtype=y_C_train.dtype)
        step_a_synth_C = np.empty((0, step_a_train.shape[1]))
        y_A_from_C     = np.empty(0, dtype=y_A_train.dtype)
        y_B_from_C     = np.empty(0, dtype=y_B_train.dtype)

    # ── Pool: originals + all synthetic rows ──────────────────────────────────
    X_final      = np.vstack([X_train,      X_synth_A,      X_synth_B,      X_synth_C])
    step_a_final = np.vstack([step_a_train, step_a_synth_A, step_a_synth_B, step_a_synth_C])
    y_A_final    = np.hstack([y_A_train,    y_A_synth,      y_A_from_B,     y_A_from_C])
    y_B_final    = np.hstack([y_B_train,    y_B_from_A,     y_B_synth,      y_B_from_C])
    y_C_final    = np.hstack([y_C_train,    y_C_from_A,     y_C_from_B,     y_C_synth])

    train_set = _SMOTEDataset(step_a_final, X_final, y_A_final, y_B_final, y_C_final)
    val_set   = SCAMAAPSequenceDataset(df_val,  encoders)
    test_set  = SCAMAAPSequenceDataset(df_test, encoders)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, encoders


if __name__ == "__main__":
    import os
    filepath = os.path.join(os.path.dirname(__file__), "scamaap dataset.csv")
    train_loader, val_loader, test_loader, encoders = get_dataloaders(filepath)

    print("Train batches:", len(train_loader))
    print("Val   batches:", len(val_loader))
    print("Test  batches:", len(test_loader))
    print("Supervisory classes:", encoders.enc_supervisory.classes_)
    print("Operator classes   :", encoders.enc_operator.classes_)
    print("Unsafe classes     :", encoders.enc_unsafe.classes_)

    s_a, s0, yA, yB, yC = next(iter(train_loader))
    print("\nstep_a shape:", s_a.shape)  # (B, 2)
    print("step0  shape:", s0.shape)    # (B, 5)
    print("y_A shape   :", yA.shape)
    print("y_B shape   :", yB.shape)
    print("y_C shape   :", yC.shape)
