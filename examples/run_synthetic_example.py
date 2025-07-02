import numpy as np
import pandas as pd
from hmm_regime_identification import HMMRegimeIdentifier, HMMType


def generate_synthetic_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Generates synthetic irregularly sampled data for HMM regime identification.

    The data simulates three regimes: mean-reverting, trend-following (up), and trend-following (down).
    Columns generated include 'price_delta' (target), 'volatility', 'time_since_last_obs',
    'moving_average_ratio', and 'rsi_proxy' (features).

    Args:
        n_samples (int): The number of data points to generate.
        random_state (int): Seed for the random number generator for reproducibility.

    Returns:
        pd.DataFrame: DataFrame with generated data, indexed by timestamp.
    """
    np.random.seed(random_state)
    data = []
    current_price = 100.0
    last_obs_time = 0

    for i in range(n_samples):
        time_elapsed = np.random.uniform(0.1, 2.0)
        current_time = last_obs_time + time_elapsed

        if i < n_samples / 3:
            # Mean-reverting regime
            prev_day_price_delta = (100 - current_price) * 0.05 + np.random.normal(0, 0.5)
            volatility = np.random.uniform(0.5, 1.5)
            moving_average_ratio = np.random.uniform(0.98, 1.02)  # Around 1 for mean-reversion
            rsi_proxy = np.random.uniform(30, 70)  # Within normal range for mean-reversion
        elif i < 2 * n_samples / 3:
            # Trend-following (up) regime
            prev_day_price_delta = np.random.normal(1.0, 0.8)
            volatility = np.random.uniform(1.0, 2.0)
            moving_average_ratio = np.random.uniform(1.01, 1.05)  # >1 for uptrend
            rsi_proxy = np.random.uniform(60, 90)  # Overbought for uptrend
        else:
            # Trend-following (down) regime
            prev_day_price_delta = np.random.normal(-1.0, 0.8)
            volatility = np.random.uniform(1.0, 2.0)
            moving_average_ratio = np.random.uniform(0.95, 0.99)  # <1 for downtrend
            rsi_proxy = np.random.uniform(10, 40)  # Oversold for downtrend

        current_price += prev_day_price_delta

        data.append(
            {
                "timestamp": current_time,
                "prev_day_price_delta": prev_day_price_delta,
                "volatility": volatility,
                "time_since_last_obs": time_elapsed,
                "moving_average_ratio": moving_average_ratio,
                "rsi_proxy": rsi_proxy,
            }
        )
        last_obs_time = current_time

    df = pd.DataFrame(data)
    df = df.set_index("timestamp")
    print("Synthetic data generated.")
    return df


if __name__ == "__main__":
    # 1. Generate Synthetic Data
    df_synthetic = generate_synthetic_data(n_samples=1000)

    # 2. Define interpret_column and feature labels
    interpret_col = "prev_day_price_delta"
    feature_cols = ["prev_day_price_delta", "volatility", "time_since_last_obs", "moving_average_ratio", "rsi_proxy"]

    # 3. Initialize HMMRegimeIdentifier with the generated data and labels
    regime_identifier = HMMRegimeIdentifier(
        df=df_synthetic,
        interpret_column=interpret_col,
        feature_labels=feature_cols,
        n_components=3,
        standardize_features=True,  # Set to False if you don't want standardization
        hmm_type=HMMType.GMM,  # Use GMMHMM
        n_mix=2,  # Number of mixtures for GMMHMM
    )

    # 4. Train, infer, interpret, and visualize
    regime_identifier.train_hmm()
    regime_identifier.infer_states()
    regime_identifier.interpret_regimes()
    regime_identifier.visualize_results()
