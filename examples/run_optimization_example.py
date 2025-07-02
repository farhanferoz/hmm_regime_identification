from hmm_regime_identification import HMMRegimeIdentifier
from examples.run_synthetic_example import generate_synthetic_data

if __name__ == "__main__":
    # Generate synthetic data
    df_synthetic = generate_synthetic_data(n_samples=1000)

    # 2. Define interpret_column and feature labels
    interpret_col = "prev_day_price_delta"
    feature_cols = ["prev_day_price_delta", "volatility", "time_since_last_obs", "moving_average_ratio", "rsi_proxy"]

    # Initialize HMMRegimeIdentifier and run optimization directly
    regime_identifier = HMMRegimeIdentifier(
        df=df_synthetic, interpret_column=interpret_col, feature_labels=feature_cols, random_state=42, run_optimization=True, optimization_n_trials=50
    )

    study = regime_identifier.study

    print("\nNumber of finished trials: ", len(study.trials))
    print("\nBest trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
