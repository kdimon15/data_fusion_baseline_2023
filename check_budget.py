#!/usr/bin/env python3

import json
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

BUDGET = 10


def check_budget(source_path, attack_patch, quantile_path):

    with open(quantile_path, "rb") as f:
        quantiles = json.load(f)

    df_source = pd.read_csv(
        source_path,
        parse_dates=["transaction_dttm"],
        dtype={"user_id": int, "mcc_code": int, "currency_rk": int, "transaction_amt": float},
    ).sort_values(by=["user_id", "transaction_dttm"])

    df_attack = pd.read_csv(
        attack_patch,
        parse_dates=["transaction_dttm"],
        dtype={"user_id": int, "mcc_code": int, "currency_rk": int, "transaction_amt": float},
    ).sort_values(by=["user_id", "transaction_dttm"])

    assert df_source.shape == df_attack.shape

    def is_different_records(a, b):
        return not all(
            [
                a.user_id == b.user_id,
                a.mcc_code == b.mcc_code,
                a.currency_rk == b.currency_rk,
                np.isclose(a.transaction_amt, b.transaction_amt),
                a.transaction_dttm == b.transaction_dttm,
            ]
        )

    diff_count = defaultdict(int)

    for a, b in tqdm(zip(df_source.itertuples(), df_attack.itertuples()), total=df_source.shape[0]):
        if is_different_records(a, b):
            diff_count[a.user_id] += 1
            if diff_count[a.user_id] > BUDGET:
                return False

            if np.sign(a.transaction_amt) != np.sign(b.transaction_amt):
                return False

            if a.transaction_amt < 0:
                ruler = quantiles["negative"]
            else:
                ruler = quantiles["positive"]
            #key_a = str(a.mcc_code)
            key_b = str(b.mcc_code)
            #upper_bound_a = ruler["max"][key_a]
            #lower_bound_a = ruler["min"][key_a]
            upper_bound_b = ruler["max"][key_b]
            lower_bound_b = ruler["min"][key_b]
            if any(
                [
                    #upper_bound_a < a.transaction_amt,
                    upper_bound_b < b.transaction_amt,
                    #lower_bound_a > a.transaction_amt,
                    lower_bound_b > b.transaction_amt,
                ]
            ):
                return False
    return True


def main():
    source_path = "../sample_submission.csv"
    attack_patch = "../naive_submission.csv"
    quantile_path = "quantiles.json"
    print(check_budget(source_path, attack_patch, quantile_path))

if __name__ == "__main__":
    main()
