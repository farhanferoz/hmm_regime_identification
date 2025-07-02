import numpy as np
import pandas as pd
from hmm_regime_identification import HMMRegimeIdentifier

if __name__ == "__main__":
    # Observed Data (Example Sequence):
    X = np.array([[-0.2], [0.4], [-1.0], [4.8], [5.2], [4.9], [5.1], [0.1], [-0.3], [0.5]])

    # Convert to DataFrame for HMMRegimeIdentifier
    df_analytical = pd.DataFrame(X, columns=["observation"])

    # Define interpret_column and feature labels
    interpret_col = "observation"
    feature_cols = ["observation"]

    # Initialize HMMRegimeIdentifier
    regime_identifier = HMMRegimeIdentifier(
        df=df_analytical,
        interpret_column=interpret_col,
        feature_labels=feature_cols,
        n_components=2,
        covariance_type="diag",
        n_iter=100,
        random_state=42,
        standardize_features=False,  # No need to standardize for this simple example
    )

    # Train, infer, and interpret
    regime_identifier.train_hmm()
    regime_identifier.infer_states()
    regime_identifier.interpret_regimes()

    # Print learned parameters and inferred states
    print("\n--- Learned Parameters ---")
    # The interpret_regimes method already prints detailed parameters, so we just need to print the inferred states here.
    print("\n--- Inferred States ---")
    print(f"States:\n{regime_identifier.df['regime'].values}")
