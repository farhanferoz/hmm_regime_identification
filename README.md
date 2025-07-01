# HMM-based Market Regime Identification

This project implements a Hidden Markov Model (HMM) to identify different market regimes (e.g., mean-reverting, trend-following/momentum) from irregularly sampled financial data. The code is structured in an object-oriented manner for modularity and reusability.

## Table of Contents
- [Features](#features)
- [How it Works](#how-it-works)
- [Technical Description of HMM](#technical-description-of-hmm)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)

## Features
- **Synthetic Data Generation:** Generates irregularly sampled data simulating different market regimes (mean-reverting, trend-following up, trend-following down).
- **Hidden Markov Model (HMM):** Utilizes `hmmlearn.GaussianHMM` to model the underlying market states.
- **Regime Identification:** Infers the most probable sequence of hidden states (regimes) from the observed data.
- **Regime Interpretation:** Attempts to label the identified regimes based on the mean of a specified target variable (e.g., price delta) within each state.
- **Visualization:** Provides plots to visualize the target variable colored by the inferred regimes and the regime transitions over time.
- **Object-Oriented Design:** The core logic is encapsulated within a `HMMRegimeIdentifier` class for better organization and reusability.

## How it Works

The project follows these main steps:

1.  **Data Generation (or Loading):**
    *   A `generate_synthetic_data` function creates a sample dataset with irregular time intervals and features (`price_delta`, `volatility`, `time_since_last_obs`, `moving_average_ratio`, `rsi_proxy`) that exhibit characteristics of different market regimes.
    *   In a real-world scenario, you would replace this with your actual irregularly sampled financial data.

2.  **HMM Initialization:**
    *   The `HMMRegimeIdentifier` class is initialized with the DataFrame, the name of the target column, a list of feature column names, and HMM parameters (number of components/regimes, covariance type, iterations, random state).
    *   It internally extracts the specified features into a NumPy array, which is the required format for the `hmmlearn` library.

3.  **HMM Training:**
    *   The `train_hmm` method uses the `fit` method of `hmmlearn.GaussianHMM` to train the model on the provided features. This step learns the HMM's parameters:
        *   **Transition Probabilities:** The likelihood of switching from one hidden state to another.
        *   **Emission Probabilities (Means and Covariances):** The statistical distribution of the observed features within each hidden state.

4.  **State Inference:**
    *   The `infer_states` method uses the Viterbi algorithm (via `model.predict()`) to determine the most probable sequence of hidden states given the observed features. These inferred states are then added back to the original DataFrame.

5.  **Regime Interpretation:**
    *   The `interpret_regimes` method analyzes the learned mean values of the features for each hidden state. It attempts to assign meaningful labels (e.g., "Mean-Reverting", "Trend-Following (Up)", "Trend-Following (Down)") to the regimes, primarily based on the mean of the specified target variable (`price_delta` in the synthetic example).

6.  **Visualization:**
    *   The `visualize_results` method generates plots to help understand the identified regimes. One plot shows the target variable over time, colored by the inferred regime, and another shows the sequence of inferred regimes over time.

## Technical Description of HMM

A Hidden Markov Model (HMM) is a statistical Markov model in which the system being modeled is assumed to be a Markov process with unobserved (hidden) states. It is particularly useful for modeling sequential data where the underlying process is not directly observable but can be inferred from a sequence of observations.

An HMM is formally defined by the following components:

*   **N (Number of Hidden States):** The set of possible hidden states, $S = \{s_1, s_2, ..., s_N\}$. In our case, these are the market regimes.
*   **M (Number of Observation Symbols):** For a `GaussianHMM`, the observations are continuous, so we describe their distribution.
*   **A (Transition Probability Matrix):** A matrix $A = \{a_{ij}\}$ where $a_{ij} = P(q_{t+1} = s_j | q_t = s_i)$ is the probability of transitioning from state $s_i$ at time $t$ to state $s_j$ at time $t+1$. This captures the dynamics of regime switching.
*   **B (Emission Probability Distribution):** For each state $s_j$, a probability distribution $b_j(O_t) = P(O_t | q_t = s_j)$ that describes the likelihood of observing a particular observation $O_t$ given that the system is in state $s_j$. For `GaussianHMM`, this is a multivariate Gaussian distribution defined by a mean vector $\mu_j$ and a covariance matrix $\Sigma_j$ for each state $s_j$.
*   **$\pi$ (Initial State Distribution):** A vector $\pi = \{\pi_i\}$ where $\pi_i = P(q_1 = s_i)$ is the probability of the model starting in state $s_i$.

Together, these parameters $\lambda = (A, B, \pi)$ define a complete HMM.

### Simple Analytical Example: The Dishonest Casino

Consider a simplified scenario: a casino uses two dice, one fair and one loaded. You can't see which die is being used, but you observe the sequence of rolls. This is a classic HMM problem.

*   **Hidden States ($S$):** \{Fair Die, Loaded Die\}
*   **Observations ($O$):** The sequence of numbers rolled (1, 2, 3, 4, 5, 6).
*   **Transition Probabilities ($A$):**
    *   $P(\text{Fair} \to \text{Fair})$: Probability the casino keeps using the fair die.
    *   $P(\text{Fair} \to \text{Loaded})$: Probability the casino switches to the loaded die.
    *   $P(\text{Loaded} \to \text{Loaded})$: Probability the casino keeps using the loaded die.
    *   $P(\text{Loaded} \to \text{Fair})$: Probability the casino switches back to the fair die.
*   **Emission Probabilities ($B$):**
    *   For the Fair Die state: $P(O_t | \text{Fair Die})$ - each number (1-6) has a 1/6 probability.
    *   For the Loaded Die state: $P(O_t | \text{Loaded Die})$ - certain numbers (e.g., 6) have a higher probability.
*   **Initial State Distribution ($\pi$):** The probability of starting with the fair die or the loaded die.

### How HMM Solves the Casino Example

HMMs are designed to solve three fundamental problems, all of which apply to the Dishonest Casino example:

1.  **Evaluation Problem (Likelihood):** Given an HMM model (i.e., all its parameters are known) and a sequence of observations (e.g., a sequence of dice rolls), what is the probability that this sequence was generated by this model? This is solved using the **Forward Algorithm**. In the casino example, if you know the properties of both dice and how often the casino switches them, you can calculate the probability of observing a specific sequence of rolls.

2.  **Decoding Problem (Most Likely State Sequence):** Given an HMM model and a sequence of observations, what is the most likely sequence of hidden states (i.e., which die was used at each roll) that produced these observations? This is solved using the **Viterbi Algorithm**. If you observe a sequence of rolls like `[6, 6, 6, 1, 2, 6]`, the Viterbi algorithm can tell you the most probable sequence of die usage (e.g., `[Loaded, Loaded, Loaded, Fair, Fair, Loaded]`). This is the problem our `HMMRegimeIdentifier` solves to infer market regimes.

3.  **Learning Problem (Parameter Estimation):** Given a sequence of observations, how can we learn the HMM parameters (transition probabilities, emission probabilities, and initial state probabilities) that best describe the observed data? This is solved using the **Baum-Welch Algorithm** (a form of the Expectation-Maximization algorithm). In the casino example, if you only observe many sequences of rolls but don't know the probabilities of the loaded die or the switching probabilities, the Baum-Welch algorithm can estimate these parameters. This is what the `train_hmm` method in our code does when it calls `model.fit(self.features)`.

### `hmmlearn.GaussianHMM`

This project uses `hmmlearn.GaussianHMM`, which is suitable for continuous observation data. For each hidden state, it assumes that the observed features follow a multivariate Gaussian distribution. The model learns:

*   **`n_components` (Number of Hidden States):** The number of distinct market regimes we assume exist.
*   **`startprob_` (Initial State Probabilities):** The probability of starting in each hidden state.
*   **`transmat_` (Transition Matrix):** A matrix where `transmat_[i, j]` is the probability of transitioning from hidden state `i` to hidden state `j`.
*   **`means_` (Mean Vectors):** For each hidden state, a vector representing the mean of the Gaussian distribution for the observed features.
*   **`covars_` (Covariance Matrices):** For each hidden state, a matrix representing the covariance of the Gaussian distribution for the observed features. The `covariance_type` parameter (`"full"`, `"diag"`, etc.) determines the structure of these matrices.

### Dealing with Irregularly Sampled Data

While standard HMMs in `hmmlearn` do not explicitly model time intervals between observations, this implementation addresses irregularly sampled data by **including `time_since_last_obs` as one of the input features**.

By incorporating `time_since_last_obs` into the feature set, the HMM's emission probabilities for each hidden state will learn the typical distribution of time intervals associated with that state, alongside other features like `price_delta` and `volatility`. For example:

*   A "mean-reverting" regime might be characterized by smaller `price_delta` values, lower `volatility`, and potentially more frequent (smaller `time_since_last_obs`) observations if the market is actively reacting to a mean.
*   A "trend-following" regime might show consistent `price_delta` in one direction, higher `volatility`, and potentially different patterns in `time_since_last_obs` depending on how the trend manifests in observation frequency.

This approach allows the HMM to implicitly leverage the information contained in the irregular sampling intervals to better distinguish between different underlying market regimes.

## Installation

To set up the project, follow these steps:

1.  **Navigate to the project directory:**
    ```bash
    cd C:\Users\farha\PycharmProjects\MLResearch\hmm_regime_identification
    ```

2.  **Create a virtual environment:**
    It's highly recommended to use a virtual environment to manage project dependencies.
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    *   **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    *   **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install dependencies:**
    The required libraries are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Once the installation is complete, you can run the main script:

```bash
.\venv\Scripts\python hmm_regime_identification.py
```

This will:
1.  Generate synthetic data.
2.  Train the HMM.
3.  Infer the hidden market regimes.
4.  Print the learned HMM parameters and inferred regime labels.
5.  Display two plots visualizing the results.

You can modify the `if __name__ == "__main__":` block in `hmm_regime_identification.py` to:
*   Adjust the number of synthetic samples (`n_samples`).
*   Change the HMM parameters (`n_components`, `covariance_type`, `n_iter`).
*   Adapt the `target_col` and `feature_cols` to your specific data if you replace the synthetic data generation with your own dataset.

## Project Structure

```
.
├── hmm_regime_identification.py
├── requirements.txt
└── venv/
```