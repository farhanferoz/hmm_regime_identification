import numpy as np
import pandas as pd
from hmmlearn import hmm
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, List

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
            price_delta = (100 - current_price) * 0.05 + np.random.normal(0, 0.5)
            volatility = np.random.uniform(0.5, 1.5)
            moving_average_ratio = np.random.uniform(0.98, 1.02) # Around 1 for mean-reversion
            rsi_proxy = np.random.uniform(30, 70) # Within normal range for mean-reversion
        elif i < 2 * n_samples / 3:
            # Trend-following (up) regime
            price_delta = np.random.normal(1.0, 0.8)
            volatility = np.random.uniform(1.0, 2.0)
            moving_average_ratio = np.random.uniform(1.01, 1.05) # >1 for uptrend
            rsi_proxy = np.random.uniform(60, 90) # Overbought for uptrend
        else:
            # Trend-following (down) regime
            price_delta = np.random.normal(-1.0, 0.8)
            volatility = np.random.uniform(1.0, 2.0)
            moving_average_ratio = np.random.uniform(0.95, 0.99) # <1 for downtrend
            rsi_proxy = np.random.uniform(10, 40) # Oversold for downtrend

        current_price += price_delta

        data.append({
            'timestamp': current_time,
            'price_delta': price_delta,
            'volatility': volatility,
            'time_since_last_obs': time_elapsed,
            'moving_average_ratio': moving_average_ratio,
            'rsi_proxy': rsi_proxy
        })
        last_obs_time = current_time

    df = pd.DataFrame(data)
    df = df.set_index('timestamp')
    print("Synthetic data generated.")
    return df

class HMMRegimeIdentifier:
    """
    A class to identify market regimes using Hidden Markov Models (HMM).

    This class encapsulates the HMM training, state inference, regime interpretation,
    and visualization functionalities.
    """
    def __init__(
        self, 
        df: pd.DataFrame, 
        target_label: str,
        feature_labels: List[str],
        n_components: int = 3, 
        covariance_type: str = "full", 
        n_iter: int = 100, 
        random_state: int = 42
    ):
        """
        Initializes the HMMRegimeIdentifier with data, target, feature labels, and HMM parameters.

        Args:
            df (pd.DataFrame): The input DataFrame containing the time-series data, including
                               target and feature columns.
            target_label (str): The column name in `df` that represents the target variable
                                (e.g., 'price_delta'). This is used for regime interpretation.
            feature_labels (List[str]): A list of column names in `df` that represent the features
                                        to be used as observations for the HMM.
            n_components (int): The number of hidden states (regimes) to identify.
            covariance_type (str): The type of covariance matrix to use for each state
                                   ('full', 'tied', 'diag', 'spherical').
            n_iter (int): The number of iterations for the EM algorithm during HMM training.
            random_state (int): Seed for the random number generator for reproducibility.
        """
        self.df: pd.DataFrame = df.copy()  # Work on a copy to avoid modifying original DataFrame
        self.target_label: str = target_label
        self.feature_labels: List[str] = feature_labels
        
        # Extract features from the DataFrame
        self.features: np.ndarray = self.df[self.feature_labels].values

        self.n_components: int = n_components
        self.covariance_type: str = covariance_type
        self.n_iter: int = n_iter
        self.random_state: int = random_state
        self.model: hmm.GaussianHMM = hmm.GaussianHMM(n_components=n_components, covariance_type=covariance_type,
                                                     n_iter=n_iter, random_state=random_state)
        self.hidden_states: np.ndarray | None = None
        self.regime_labels: Dict[int, str] = {}

    def train_hmm(self) -> None:
        """
        Trains the Hidden Markov Model using the provided features.

        Raises:
            ValueError: If features are empty or not set.
        """
        if self.features is None or len(self.features) == 0:
            raise ValueError("Features are empty or not set.")
        print("Training HMM...")
        self.model.fit(self.features)
        print("HMM training complete.")

    def infer_states(self) -> None:
        """
        Infers the most likely sequence of hidden states (regimes) from the observed features.

        The inferred states are added as a 'regime' column to the DataFrame.

        Raises:
            ValueError: If features are empty or not set.
        """
        if self.features is None or len(self.features) == 0:
            raise ValueError("Features are empty or not set.")
        print("Inferring hidden states...")
        self.hidden_states = self.model.predict(self.features)
        self.df['regime'] = self.hidden_states
        print("Hidden states inferred.")

    def interpret_regimes(self) -> None:
        """
        Interprets the learned HMM regimes based on their mean of the target variable.

        Assigns labels like "Mean-Reverting", "Trend-Following (Up)", and "Trend-Following (Down)"
        to the inferred regimes and adds a 'regime_label' column to the DataFrame.

        Raises:
            ValueError: If hidden states have not been inferred yet.
        """
        if self.hidden_states is None:
            raise ValueError("Hidden states not inferred. Call infer_states first.")

        print("\nLearned HMM Parameters:")
        for i in range(self.model.n_components):
            print(f"\nRegime {i}:")
            # Print means for all features for clarity
            mean_values = ", ".join([f"{label}: {mean:.4f}" for label, mean in zip(self.feature_labels, self.model.means_[i])])
            print(f"  Mean ({mean_values})")
            print(f"  Covariance:\n{self.model.covars_[i]}")
            print(f"  Transition probabilities from Regime {i}: {self.model.transmat_[i]}")

        # Find the index of the target_label within the feature_labels list
        try:
            target_feature_idx = self.feature_labels.index(self.target_label)
        except ValueError:
            print(f"Warning: Target label '{self.target_label}' not found in feature_labels. Cannot interpret regimes based on target.")
            return

        # Map regimes to meaningful labels based on their means of the target feature
        mean_reverting_idx = np.argmin(np.abs(self.model.means_[:, target_feature_idx]))
        self.regime_labels[mean_reverting_idx] = "Mean-Reverting"

        remaining_indices = [i for i in range(self.n_components) if i != mean_reverting_idx]

        if len(remaining_indices) > 0:
            remaining_sorted_by_delta = sorted(remaining_indices, key=lambda x: self.model.means_[x, target_feature_idx])

            if self.model.means_[remaining_sorted_by_delta[0], target_feature_idx] < 0:
                self.regime_labels[remaining_sorted_by_delta[0]] = "Trend-Following (Down)"
            else:
                self.regime_labels[remaining_sorted_by_delta[0]] = "Neutral/Other"

            if len(remaining_indices) > 1:
                if self.model.means_[remaining_sorted_by_delta[1], target_feature_idx] > 0:
                    self.regime_labels[remaining_sorted_by_delta[1]] = "Trend-Following (Up)"
                else:
                    self.regime_labels[remaining_sorted_by_delta[1]] = "Neutral/Other"

        print("\nInferred Regime Labels:")
        for original_idx, label in self.regime_labels.items():
            print(f"  Original Regime {original_idx}: {label}")

        self.df['regime_label'] = self.df['regime'].map(self.regime_labels)

    def visualize_results(self) -> None:
        """
        Visualizes the inferred HMM regimes.

        Generates two plots:
        1. Scatter plot of the target variable over time, colored by inferred regime.
        2. Step plot of inferred regime index over time.

        Raises:
            ValueError: If data or inferred regimes are not available.
        """
        if self.df is None or 'regime' not in self.df.columns:
            raise ValueError("Data or inferred regimes not available. Run train_hmm and infer_states first.")

        plt.figure(figsize=(15, 7))
        plt.scatter(self.df.index, self.df[self.target_label], c=self.df['regime'], cmap='viridis', s=10, alpha=0.7)
        plt.title(f'{self.target_label.replace("_", " ").title()} with Inferred HMM Regimes')
        plt.xlabel('Time')
        plt.ylabel(self.target_label.replace("_", " ").title())
        plt.colorbar(label='Regime')
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(15, 5))
        plt.plot(self.df.index, self.df['regime'], drawstyle='steps-post', label='Inferred Regime')
        plt.title('Inferred HMM Regimes Over Time')
        plt.xlabel('Time')
        plt.ylabel('Regime Index')
        plt.yticks(range(self.n_components))
        plt.grid(True)
        plt.legend()
        plt.show()

        print("\nFirst few rows with inferred regimes:")
        print(self.df.head())

if __name__ == "__main__":
    # 1. Generate Synthetic Data
    df_synthetic = generate_synthetic_data(n_samples=1000)

    # 2. Define target and feature labels
    target_col = 'price_delta'
    feature_cols = ['price_delta', 'volatility', 'time_since_last_obs', 'moving_average_ratio', 'rsi_proxy']

    # 3. Initialize HMMRegimeIdentifier with the generated data and labels
    regime_identifier = HMMRegimeIdentifier(
        df=df_synthetic,
        target_label=target_col,
        feature_labels=feature_cols,
        n_components=3
    )

    # 4. Train, infer, interpret, and visualize
    regime_identifier.train_hmm()
    regime_identifier.infer_states()
    regime_identifier.interpret_regimes()
    regime_identifier.visualize_results()