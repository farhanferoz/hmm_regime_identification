import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM, GMMHMM
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from enum import Enum
import optuna
from typing import Dict, List, Optional


class HMMType(Enum):
    GAUSSIAN = "gaussian"
    GMM = "gmm"


class HMMRegimeIdentifier:
    """
    A class to identify market regimes using Hidden Markov Models (HMM).

    This class encapsulates the HMM training, state inference, regime interpretation,
    and visualization functionalities.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        interpret_column: str,
        feature_labels: List[str],
        n_components: int = 3,
        covariance_type: str = "full",
        n_iter: int = 100,
        random_state: int = 42,
        standardize_features: bool = True,
        hmm_type: HMMType = HMMType.GAUSSIAN,  # 'gaussian' or 'gmm'
        n_mix: int = 1,  # Number of mixtures for GMMHMM
        run_optimization: bool = False,
        optimization_n_trials: int = 50,
    ):
        """
        Initializes the HMMRegimeIdentifier with data, target, feature labels, and HMM parameters.

        Args:
            df (pd.DataFrame): The input DataFrame containing the time-series data, including
                               target and feature columns.
            interpret_column (str): The column name in `df` that represents the variable
                                    used for interpreting the regimes (e.g., 'prev_day_price_delta').
            feature_labels (List[str]): A list of column names in `df` that represent the features
                                        to be used as observations for the HMM.
            n_components (int): The number of hidden states (regimes) to identify.
            covariance_type (str): The type of covariance matrix to use for each state
                                   ('full', 'tied', 'diag', 'spherical').
            n_iter (int): The number of iterations for the EM algorithm during HMM training.
            random_state (int): Seed for the random number generator for reproducibility.
            standardize_features (bool): Whether to standardize the features (mean=0, variance=1)
                                         before training the HMM. Defaults to True.
            hmm_type (HMMType): The type of HMM to use. Can be `HMMType.GAUSSIAN` for `GaussianHMM` or `HMMType.GMM` for `GMMHMM`. Defaults to `HMMType.GAUSSIAN`.
            n_mix (int): The number of mixture components for each state in `GMMHMM`. Only applicable when `hmm_type` is `HMMType.GMM`. Defaults to 1.
        """
        self.df: pd.DataFrame = df.copy()  # Work on a copy to avoid modifying original DataFrame
        self.interpret_column: str = interpret_column
        self.feature_labels: List[str] = feature_labels
        self.random_state: int = random_state
        self.standardize_features: bool = standardize_features
        self.hidden_states: Optional[np.ndarray] = None
        self.regime_labels: Dict[int, str] = {}
        self.scaler: Optional[StandardScaler] = None
        self.model = None
        self.study = None

        # Extract features from the DataFrame
        features_raw: np.ndarray = self.df[self.feature_labels].values

        if self.standardize_features:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(features_raw)
            print("Features standardized.")
        else:
            self.features = features_raw

        if run_optimization:
            print(f"Running hyperparameter optimization for {optimization_n_trials} trials...")
            self.study = self.optimize_hyperparameters(n_trials=optimization_n_trials)
            best_params = self.study.best_params
            self.n_components = best_params["n_components"]
            self.covariance_type = best_params["covariance_type"]
            self.n_iter = best_params["n_iter"]
            self.hmm_type = best_params["hmm_type"]
            self.n_mix = best_params["n_mix"] if "n_mix" in best_params else 1
            print("Hyperparameter optimization complete. Initializing HMM with best parameters.")
        else:
            self.n_components = n_components
            self.covariance_type = covariance_type
            self.n_iter = n_iter
            self.hmm_type = hmm_type
            self.n_mix = n_mix

        if self.hmm_type == HMMType.GAUSSIAN:
            self.model = GaussianHMM(n_components=self.n_components, covariance_type=self.covariance_type, n_iter=self.n_iter, random_state=self.random_state)
        elif self.hmm_type == HMMType.GMM:
            self.model = GMMHMM(n_components=self.n_components, n_mix=self.n_mix, covariance_type=self.covariance_type, n_iter=self.n_iter, random_state=self.random_state)
        else:
            raise ValueError("Invalid hmm_type. Must be `HMMType.GAUSSIAN` or `HMMType.GMM`.")

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
        self.df["regime"] = self.hidden_states
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
            if self.hmm_type == HMMType.GAUSSIAN:
                # For GaussianHMM, means_[i] is a 1D array of feature means for state i
                mean_values = ", ".join([f"{label}: {mean:.4f}" for label, mean in zip(self.feature_labels, self.model.means_[i])])
                print(f"  Mean ({mean_values})")
            elif self.hmm_type == HMMType.GMM:
                # For GMMHMM, means_[i] is a 2D array (n_mix, n_features)
                print(f"  Means (Mixture Components):")
                for mix_idx, mix_mean in enumerate(self.model.means_[i]):
                    mean_values = ", ".join([f"{label}: {mean:.4f}" for label, mean in zip(self.feature_labels, mix_mean)])
                    print(f"    Component {mix_idx}: ({mean_values})")

            print(f"  Covariance:\n{self.model.covars_[i]}")
            print(f"  Transition probabilities from Regime {i}: {self.model.transmat_[i]}")

        # For regime interpretation, we need a single mean for each state.
        if self.hmm_type == HMMType.GAUSSIAN:
            state_means_for_interpretation = self.model.means_
        elif self.hmm_type == HMMType.GMM:
            # For GMMHMM, self.model.means_ is (n_components, n_mix, n_features)
            # We use the mean of the first mixture component for each state for simplicity in interpretation.
            state_means_for_interpretation = self.model.means_[:, 0, :]

        # Find the index of the interpret_column within the feature_labels list
        try:
            interpret_column_idx = self.feature_labels.index(self.interpret_column)
        except ValueError:
            print(f"Warning: Interpretation column '{self.interpret_column}' not found in feature_labels. Cannot interpret regimes based on this column.")
            return

        # Map regimes to meaningful labels based on their means of the interpret_column
        mean_reverting_idx = np.argmin(np.abs(state_means_for_interpretation[:, interpret_column_idx]))
        self.regime_labels[mean_reverting_idx] = "Mean-Reverting"

        remaining_indices = [i for i in range(self.n_components) if i != mean_reverting_idx]

        if len(remaining_indices) > 0:
            remaining_sorted_by_delta = sorted(remaining_indices, key=lambda x: state_means_for_interpretation[x, interpret_column_idx])

            if state_means_for_interpretation[remaining_sorted_by_delta[0], interpret_column_idx] < 0:
                self.regime_labels[remaining_sorted_by_delta[0]] = "Trend-Following (Down)"
            else:
                self.regime_labels[remaining_sorted_by_delta[0]] = "Neutral/Other"

            if len(remaining_indices) > 1:
                if state_means_for_interpretation[remaining_sorted_by_delta[1], interpret_column_idx] > 0:
                    self.regime_labels[remaining_sorted_by_delta[1]] = "Trend-Following (Up)"
                else:
                    self.regime_labels[remaining_sorted_by_delta[1]] = "Neutral/Other"

        print("\nInferred Regime Labels:")
        for original_idx, label in self.regime_labels.items():
            print(f"  Original Regime {original_idx}: {label}")

        self.df["regime_label"] = self.df["regime"].map(self.regime_labels)

    def visualize_results(self) -> None:
        """
        Visualizes the inferred HMM regimes.

        Generates two plots:
        1. Scatter plot of the target variable over time, colored by inferred regime.
        2. Step plot of inferred regime index over time.

        Raises:
            ValueError: If data or inferred regimes are not available.
        """
        if self.df is None or "regime" not in self.df.columns:
            raise ValueError("Data or inferred regimes not available. Run train_hmm and infer_states first.")

        plt.figure(figsize=(15, 7))
        plt.scatter(self.df.index, self.df[self.interpret_column], c=self.df["regime"], cmap="viridis", s=10, alpha=0.7)
        plt.title(f'{self.interpret_column.replace("_", " ").title()} with Inferred HMM Regimes')
        plt.xlabel("Time")
        plt.ylabel(self.interpret_column.replace("_", " ").title())
        plt.colorbar(label="Regime")
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(15, 5))
        plt.plot(self.df.index, self.df["regime"], drawstyle="steps-post", label="Inferred Regime")
        plt.title("Inferred HMM Regimes Over Time")
        plt.xlabel("Time")
        plt.ylabel("Regime Index")
        plt.yticks(range(self.n_components))
        plt.grid(True)
        plt.legend()
        plt.show()

        print("\nFirst few rows with inferred regimes:")
        print(self.df.head())

    def optimize_hyperparameters(self, n_trials: int = 50) -> optuna.Study:
        """
        Optimizes HMM hyperparameters using Optuna.

        Args:
            n_trials (int): The number of trials for Optuna optimization.

        Returns:
            optuna.Study: The Optuna study object containing optimization results.
        """

        def objective(trial: optuna.Trial) -> float:
            # Hyperparameters to optimize
            n_components = trial.suggest_int("n_components", 2, 5)
            covariance_type = trial.suggest_categorical("covariance_type", ["spherical", "diag", "full"])
            n_iter = trial.suggest_int("n_iter", 50, 200)
            hmm_type = trial.suggest_categorical("hmm_type", [HMMType.GAUSSIAN, HMMType.GMM])

            n_mix = 1
            if hmm_type == HMMType.GMM:
                n_mix = trial.suggest_int("n_mix", 1, 3)

            try:
                # Create a temporary HMMRegimeIdentifier instance for the trial
                temp_regime_identifier = HMMRegimeIdentifier(
                    df=self.df.copy(),  # Use a copy of the original DataFrame
                    interpret_column=self.interpret_column,
                    feature_labels=self.feature_labels,
                    n_components=n_components,
                    covariance_type=covariance_type,
                    n_iter=n_iter,
                    random_state=self.random_state,  # Use the same random state for reproducibility
                    standardize_features=self.standardize_features,
                    hmm_type=hmm_type,
                    n_mix=n_mix,
                    run_optimization=False,  # Prevent recursive optimization
                )

                temp_regime_identifier.train_hmm()
                # The score method returns the log-likelihood of the model
                return temp_regime_identifier.model.score(temp_regime_identifier.features)

            except Exception as e:
                print(f"Trial failed with error: {e}")
                return -np.inf  # Return negative infinity for failed trials

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        return study
