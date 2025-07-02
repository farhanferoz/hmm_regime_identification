# HMM-based Market Regime Identification

This project implements a Hidden Markov Model (HMM) to identify different market regimes (e.g., mean-reverting, trend-following/momentum) from irregularly sampled financial data. The code is structured in an object-oriented manner for modularity and reusability.

## Table of Contents
- [Features](#features)
- [How it Works](#how-it-works)
- [Technical Description of HMM](#technical-description-of-hmm)
- [Illustrative Examples](#illustrative-examples)
  - [The Dishonest Casino Example](#the-dishonest-casino-example)
    - [How HMM Solves the Casino Example](#how-hmm-solves-the-casino-example)
    - [Why the Learned Parameters May Still Differ](#why-the-learned-parameters-may-still-differ)
  - [Gaussian HMM Analytical Example](#gaussian-hmm-analytical-example)
    - [Why the Learned Covariances May Be Inaccurate](#why-the-learned-covariances-may-be-inaccurate)
  - [How to Choose the Number of Iterations (n_iter)](#how-to-choose-the-number-of-iterations-n_iter)
- [hmmlearn.GaussianHMM](#hmmlearn-gaussianhmm)
- [hmmlearn.GMMHMM](#hmmlearn-gmmhmm)
- [When to choose HMMType.GAUSSIAN vs. HMMType.GMM?](#when-to-choose-hmmtype-gaussian-vs-hmmtype-gmm)
- [How to choose the number of mixture components (n_mix) for HMMType.GMM?](#how-to-choose-the-number-of-mixture-components-n_mix-for-hmmtype-gmm)
- [Understanding the Interpretation Column: interpret_column](#understanding-the-interpretation-column-interpret_column)
- [Dealing with Irregularly Sampled Data](#dealing-with-irregularly-sampled-data)
- [Feature Standardization](#feature-standardization)
- [Scalability Considerations](#scalability-considerations)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Synthetic Data Example](#running-the-synthetic-data-example)
  - [Running the Analytical Example](#running-the-analytical-example)
  - [Running the Dishonest Casino Example](#running-the-dishonest-casino-example)
  - [Hyperparameter Optimization with Optuna](#hyperparameter-optimization-with-optuna)
- [Project Structure](#project-structure)

## Features
- **Synthetic Data Generation:** Generates irregularly sampled data simulating different market regimes (mean-reverting, trend-following up, trend-following down).
- **Hidden Markov Model (HMM):** Utilizes `hmmlearn.GaussianHMM` and `hmmlearn.GMMHMM` to model the underlying market states.
- **Regime Identification:** Infers the most probable sequence of hidden states (regimes) from the observed data.
- **Regime Interpretation:** Attempts to label the identified regimes based on the mean of a specified interpretation column (e.g., `prev_day_price_delta`).
- **Visualization:** Provides plots to visualize the interpretation column colored by the inferred regimes and the regime transitions over time.
- **Object-Oriented Design:** The core logic is encapsulated within a `HMMRegimeIdentifier` class for better organization and reusability.
- **Hyperparameter Optimization:** Integrates Optuna for automated hyperparameter tuning of the HMM.

## How it Works

The project follows these main steps:

1.  **Data Generation (or Loading):**
    *   A `generate_synthetic_data` function creates a sample dataset with irregular time intervals and features (`prev_day_price_delta`, `volatility`, `time_since_last_obs`, `moving_average_ratio`, `rsi_proxy`) that exhibit characteristics of different market regimes.
    *   In a real-world scenario, you would replace this with your actual irregularly sampled financial data.

2.  **HMM Initialization:**
    *   The `HMMRegimeIdentifier` class is initialized with the DataFrame, the name of the column used for interpretation, a list of feature column names, and HMM parameters (number of components/regimes, covariance type, iterations, random state, HMM type (`HMMType.GAUSSIAN` or `HMMType.GMM`), and number of mixtures (`n_mix`) for `GMMHMM`).
    *   It internally extracts the specified features into a NumPy array, which is the required format for the `hmmlearn` library.

    **Parameters for `HMMRegimeIdentifier`:**
    *   `df` (pd.DataFrame): The input DataFrame containing the time-series data, including interpretation and feature columns.
    *   `interpret_column` (str): The column name in `df` that represents the variable used for interpreting the regimes (e.g., 'prev_day_price_delta').
    *   `feature_labels` (List[str]): A list of column names in `df` that represent the features to be used as observations for the HMM.
    *   `n_components` (int): The number of hidden states (regimes) to identify.
    *   `covariance_type` (str): The type of covariance matrix to use for each state ('full', 'tied', 'diag', 'spherical').
    *   `n_iter` (int): The maximum number of iterations for the EM algorithm during HMM training. The algorithm stops if convergence is reached before this number.
    *   `random_state` (int): Seed for the random number generator for reproducibility.
    *   `standardize_features` (bool): Whether to standardize the features (mean=0, variance=1) before training the HMM. Defaults to `True`.
    *   `hmm_type` (HMMType): The type of HMM to use. Can be `HMMType.GAUSSIAN` for `GaussianHMM` or `HMMType.GMM` for `GMMHMM`. Defaults to `HMMType.GAUSSIAN`.
    *   `n_mix` (int): The number of mixture components for each state in `GMMHMM`. Only applicable when `hmm_type` is `HMMType.GMM`. Defaults to 1.
    *   `run_optimization` (bool): If `True`, the `HMMRegimeIdentifier` will perform hyperparameter optimization using Optuna during initialization to find the best HMM parameters. Defaults to `False`.
    *   `optimization_n_trials` (int): The number of trials Optuna will run if `run_optimization` is `True`. Defaults to 50.

    **Methods:**
    *   `train_hmm()`: Trains the HMM model.
    *   `infer_states()`: Infers the most likely sequence of hidden states.
    *   `interpret_regimes()`: Interprets and labels the identified regimes.
    *   `visualize_results()`: Generates plots to visualize the results.
    *   `optimize_hyperparameters(n_trials: int = 50)`: Optimizes HMM hyperparameters using Optuna. Returns an Optuna `Study` object. This method can be called directly on an `HMMRegimeIdentifier` instance, and it will use the `df`, `interpret_column`, `feature_labels`, and `random_state` from the instance while optimizing other HMM parameters.

3.  **HMM Training:**
    *   The `train_hmm` method uses the `fit` method of `hmmlearn.GaussianHMM` (or `GMMHMM`) to train the model on the provided features. This step learns the HMM's parameters:
        *   **Transition Probabilities:** The likelihood of switching from one hidden state to another.
        *   **Emission Probabilities (Means and Covariances):** The statistical distribution of the observed features within each hidden state.

4.  **State Inference:**
    *   The `infer_states` method uses the Viterbi algorithm (via `model.predict()`) to determine the most probable sequence of hidden states given the observed features. These inferred states are then added back to the original DataFrame.

5.  **Regime Interpretation:**
    *   The `interpret_regimes` method analyzes the learned mean values of the features for each hidden state. It attempts to assign meaningful labels (e.g., "Mean-Reverting", "Trend-Following (Up)", "Trend-Following (Down)") to the regimes, primarily based on the mean of the specified interpretation column (`prev_day_price_delta` in the synthetic example).

6.  **Visualization:**
    *   The `visualize_results` method generates plots to help understand the identified regimes. One plot shows the interpretation column over time, colored by the inferred regime, and another shows the sequence of inferred regimes over time.

## Technical Description of HMM

A Hidden Markov Model (HMM) is a statistical Markov model in which the system being modeled is assumed to be a Markov process with unobserved (hidden) states. It is particularly useful for modeling sequential data where the underlying process is not directly observable but can be inferred from a sequence of observations.

An HMM is formally defined by the following components:

*   **N (Number of Hidden States):** The set of possible hidden states, $S = \{s_1, s_2, ..., s_N\}$. In our case, these are the market regimes.
*   **M (Number of Observation Symbols):** For a `GaussianHMM`, the observations are continuous, so we describe their distribution.
*   **A (Transition Probability Matrix):** A matrix $A = \{a_{ij}\}$ where $a_{ij} = P(q_{t+1} = s_j | q_t = s_i)$ is the probability of transitioning from state $s_i$ at time $t$ to state $s_j$ at time $t+1$. This captures the dynamics of regime switching.
*   **B (Emission Probability Distribution):** For each state $s_j$, a probability distribution $b_j(O_t) = P(O_t | q_t = s_j)$ that describes the likelihood of observing a particular observation $O_t$ given that the system is in state $s_j$. For `GaussianHMM`, this is a multivariate Gaussian distribution defined by a mean vector $\mu_j$ and a covariance matrix $\Sigma_j$ for each state $s_j$.
*   **$\pi$ (Initial State Distribution):** A vector $\pi = \{\pi_i\}$ where $\pi_i = P(q_1 = s_i)$ is the probability of the model starting in state $s_i$.

Together, these parameters $\lambda = (A, B, \pi)$ define a complete HMM.

## Illustrative Examples

### The Dishonest Casino Example

Consider a simplified scenario: a casino uses two dice, one fair and one loaded. You can't see which die is being used, but you observe the sequence of rolls. This is a classic HMM problem.

**Setup:**

For the simulated data in `examples/run_casino_example.py`, the following parameters are used:

*   **Number of Rolls:** 10000
*   **Fair Die Probability:** 1/6 for each face (standard die)
*   **Loaded Die Probability:** 0.5 for rolling a 6, and the remaining 0.5 probability distributed evenly among 1-5.
*   **Switch Probability:** 0.05 (5% chance of switching between fair and loaded die at each step).

*   **Hidden States ($S$):** {Fair Die, Loaded Die}
*   **Observations ($O$):** The sequence of numbers rolled (1, 2, 3, 4, 5, 6).
*   **Transition Probabilities ($A$):
    *   $P(\text{Fair} \to \text{Fair})$: Probability the casino keeps using the fair die.
    *   $P(\text{Fair} \to \text{Loaded})$: Probability the casino switches to the loaded die.
    *   $P(\text{Loaded} \to \text{Loaded})$: Probability the casino keeps using the loaded die.
    *   $P(\text{Loaded} \to \text{Fair})$: Probability the casino switches back to the fair die.
*   **Emission Probabilities ($B$):
    *   For the Fair Die state: $P(O_t | \text{Fair Die})$ - each number (1-6) has a 1/6 probability.
    *   For the Loaded Die state: $P(O_t | \text{Loaded Die})$ - certain numbers (e.g., 6) have a higher probability.
*   **Initial State Distribution ($\pi$):** The probability of starting with the fair die or the loaded die.

### How HMM Solves the Casino Example

HMMs are designed to solve three fundamental problems, all of which apply to the Dishonest Casino example:

1.  **Evaluation Problem (Likelihood):** Given an HMM model (i.e., all its parameters are known) and a sequence of observations (e.g., a sequence of dice rolls), what is the probability that this sequence was generated by this model? This is solved using the **Forward Algorithm**. In the casino example, if you know the properties of both dice and how often the casino switches them, you can calculate the probability of observing a specific sequence of rolls.

2.  **Decoding Problem (Most Likely State Sequence):** Given an HMM model and a sequence of observations, what is the most likely sequence of hidden states (i.e., which die was used at each roll) that produced these observations? This is solved using the **Viterbi Algorithm**. If you observe a sequence of rolls like `[6, 6, 6, 1, 2, 6]`, the Viterbi algorithm can tell you the most probable sequence of die usage (e.g., `[Loaded, Loaded, Loaded, Fair, Fair, Loaded]`). This is the problem our `HMMRegimeIdentifier` solves to infer market regimes.

3.  **Learning Problem (Parameter Estimation):** Given a sequence of observations, how can we learn the HMM parameters (transition probabilities, emission probabilities, and initial state probabilities) that best describe the observed data? This is solved using the **Baum-Welch Algorithm** (a form of the Expectation-Maximization algorithm). In the casino example, if you only observe many sequences of rolls but don't know the probabilities of the loaded die or the switching probabilities, the Baum-Welch algorithm can estimate these parameters. This is what the `train_hmm` method in our code does when it calls `model.fit(self.features)`.

**Actual Results from Running the Example Code:**

The code for this analytical example can be found in `examples/run_casino_example.py`. Running this script produces the following output, which demonstrates the HMM's ability to learn the underlying parameters from the data alone:

```
Simulating Dishonest Casino Data...
Generated 10000 observations.
Training HMM...
HMM training complete.
Inferring hidden states...
Hidden states inferred.

Learned HMM Parameters:

Regime 0:
  Mean (roll: 3.0304)
  Covariance:
[[2.00077836]]
  Transition probabilities from Regime 0: [0.69899727 0.30100273]

Regime 1:
  Mean (roll: 6.0000)
  Covariance:
[[2.97698702e-06]]
  Transition probabilities from Regime 1: [0.59498856 0.40501144]

Inferred Regime Labels:
  Original Regime 0: Mean-Reverting
  Original Regime 1: Neutral/Other

Accuracy of inferred states: 0.67
```

**Comparison of True vs. Learned Parameters (Casino Example):**

| Parameter             | True Value (Casino)                               | Learned Value (Casino)                               |
|-----------------------|---------------------------------------------------|------------------------------------------------------|
| **State 0 Mean**      | ~3.5 (Fair Die)                                   | 3.0304 (Inferred State 0)                            |
| **State 1 Mean**      | ~4.5 (Loaded Die)                                 | 6.0000 (Inferred State 1)                            |
| **State 0 Covariance**| Small (Fair Die)                                  | [[2.0008]] (Inferred State 0)                        |
| **State 1 Covariance**| Larger (Loaded Die)                               | [[0.0000]] (Inferred State 1)                        |
| **Transition 0->0**   | 0.95 (Fair->Fair)                                 | 0.6990 (Inferred State 0->0)                         |
| **Transition 0->1**   | 0.05 (Fair->Loaded)                               | 0.3010 (Inferred State 0->1)                         |
| **Transition 1->0**   | 0.05 (Loaded->Fair)                               | 0.5950 (Inferred State 1->0)                         |
| **Transition 1->1**   | 0.95 (Loaded->Loaded)                             | 0.4050 (Inferred State 1->1)                         |

*Note: The mapping of inferred states (0, 1) to conceptual states (Fair/Loaded) is based on the proximity of their learned means to the true means.*

#### Why the Learned Parameters May Still Differ

Even with a large number of rolls (10,000), the learned parameters for the casino problem, particularly the mean of the loaded die state and the transition probabilities, may still not perfectly match the true values. Here's why:

1.  **Model Mismatch (Gaussian vs. Discrete):**
    *   The true data (die rolls) are **discrete integers** (1-6).
    *   We are using `hmmlearn.GaussianHMM`, which assumes that observations within each hidden state follow a **continuous Gaussian distribution**. This is an inherent approximation. The model tries to fit a continuous bell curve to discrete values. While it can distinguish between states, it cannot perfectly replicate the exact probabilities of discrete outcomes.

2.  **Loaded Die's Non-Gaussian Emission:**
    *   The loaded die has a very specific, **non-Gaussian emission distribution** (50% chance of rolling a 6, and the remaining 50% distributed among 1-5). A single Gaussian distribution (as used by `GaussianHMM`) is a poor fit for such a distribution.
    *   The HMM attempts to find the "best" single Gaussian to represent this, which results in a mean (e.g., 6.0000) and a very small covariance (near zero) for the loaded die state. This indicates the model is trying to concentrate the probability mass heavily around the value 6, which is the most frequent outcome for the loaded die.
    *   This imperfect modeling of the emission distribution can influence the learned transition probabilities, as the model tries to compensate for the emission mismatch by adjusting state transitions.

3.  **Heuristic State Mapping:**
    *   Our interpretation relies on a heuristic mapping of the HMM's arbitrary internal state labels (0, 1) to our conceptual "Fair" and "Loaded" states based on the learned means. While this mapping is generally effective, it's an external step and doesn't change the HMM's internal approximations.
 
4. **`GMMHMM` Instability for Simple Discrete Data:**
   *   While `GMMHMM` is theoretically better suited for multimodal distributions, in practice, for very simple discrete data like die rolls with only one feature, attempting to fit multiple Gaussian components (`n_mix > 1`) can lead to numerical instability and convergence issues (as seen when trying to run the example with `GMMHMM`). The model might struggle to find meaningful distinct Gaussian components for such a limited and discrete observation space.

In summary, while increasing the data size significantly improves the HMM's ability to learn, the fundamental mismatch between the `GaussianHMM`'s continuous assumption and the discrete, non-Gaussian nature of the casino problem's emissions prevents a perfect recovery of the true parameters. For a more accurate fit, an HMM with discrete emission probabilities (if available in `hmmlearn` or a custom implementation) would be more appropriate.

### Gaussian HMM Analytical Example

Let's consider a simplified analytical example to illustrate how `hmmlearn.GaussianHMM` works with continuous observations.

**Setup:**

Suppose we have a process with two hidden states, and we observe a sequence of numbers. We don't know which state generated which number, but we assume:

*   **Hidden States:**
    *   **State 0:** Emits observations drawn from a Gaussian distribution with **mean $\mu_0 = 0$ and standard deviation $\sigma_0 = 1$** (i.e., $N(0, 1)$).
    *   **State 1:** Emits observations drawn from a Gaussian distribution with **mean $\mu_1 = 5$ and standard deviation $\sigma_1 = 1$** (i.e., $N(5, 1)$).

*   **Initial State Probabilities ($\pi$):** The probability of starting in State 0 or State 1 is equal: $\pi = [0.5, 0.5]$.

*   **Transition Matrix ($A$):** The probabilities of switching between states are:
    ```
    A = [[0.9, 0.1],  // From State 0: 90% chance to stay in State 0, 10% to go to State 1
         [0.1, 0.9]]  // From State 1: 10% chance to go to State 0, 90% to stay in State 1
    ```

**Observed Data (Example Sequence):**

Let's say we observe the following sequence of numbers:

`X = [-0.2, 0.4, -1.0, 4.8, 5.2, 4.9, 5.1, 0.1, -0.3, 0.5]`

**How HMM Solves This:**

Given only the observed data `X`, an HMM can infer the underlying hidden states and learn the model parameters.

1.  **Learning (Parameter Estimation):** The `hmmlearn.GaussianHMM` model, when `fit()` to `X`, will use the Baum-Welch Algorithm to estimate the parameters. It will try to find the means, covariances, and transition probabilities that best explain the observed sequence. Ideally, it would learn parameters close to:
    *   **Means:** One mean close to 0, another close to 5.
    *   **Covariances:** Both covariances close to 1.
    *   **Transition Matrix:** A matrix similar to the one defined above, showing persistence within states.

2.  **Decoding (Most Likely State Sequence):** After learning the parameters, the `predict()` method (which uses the Viterbi algorithm) can be used to infer the most likely sequence of hidden states that generated `X`. For the given `X`, the HMM would likely output a sequence like:
    
    `states = [0, 0, 0, 1, 1, 1, 1, 0, 0, 0]`

**Important Note on State Labeling:**
The HMM algorithm assigns arbitrary integer labels (e.g., 0, 1, 2...) to the hidden states it discovers. These labels do not inherently correspond to the conceptual labels (e.g., "State 0: N(0,1)", "State 1: N(5,1)") we use in our setup. We must interpret the learned parameters (especially the means) to map the HMM's internal labels to our conceptual states.

**Actual Results from Running the Example Code:**

The code for this analytical example can be found in `examples/run_analytical_example.py`. Running this script produces the following output, which demonstrates the HMM's ability to learn the underlying parameters from the data alone:

```
--- Learned Parameters ---
Means:
[[ 5.        ]  # This corresponds to our conceptual State 1 (mean ~5)
 [-0.08333333]] # This corresponds to our conceptual State 0 (mean ~0)
Covariances:
[[[0.0275    ]]  # Covariance for the state with mean ~5
 [[0.25305556]]] # Covariance for the state with mean ~0
Transition Matrix:
[[0.75 0.25]  # From internal state 0 (conceptual State 1)
 [0.2  0.8 ]]  # From internal state 1 (conceptual State 0)

--- Inferred States ---
States:
[1 1 1 0 0 0 0 1 1 1]
```
**Interpretation of Inferred States:**
Based on the learned means, the HMM's internal State 0 corresponds to our conceptual State 1 (mean ~5), and the HMM's internal State 1 corresponds to our conceptual State 0 (mean ~0). Therefore, the inferred state sequence `[1 1 1 0 0 0 0 1 1 1]` actually maps to:
`[State 0, State 0, State 0, State 1, State 1, State 1, State 1, State 0, State 0, State 0]`
This sequence is consistent with the true underlying states that generated the data.

This demonstrates how the HMM can successfully uncover the hidden process (which state was active at each observation) from the observable data, even without prior knowledge of the state parameters.

**Comparison of True vs. Learned Parameters (Analytical Example):**

| Parameter             | True Value (Analytical) | Learned Value (Analytical) |
|-----------------------|-------------------------|----------------------------|
| **State 0 Mean**      | 0.0                     | -0.0833                    |
| **State 1 Mean**      | 5.0                     | 5.0                        |
| **State 0 Covariance**| [[1.0]]                 | [[0.2531]]                 |
| **State 1 Covariance**| [[1.0]]                 | [[0.0275]]                 |
| **Transition 0->0**   | 0.9                     | 0.8                        |
| **Transition 0->1**   | 0.1                     | 0.2                        |
| **Transition 1->0**   | 0.1                     | 0.25                       |
| **Transition 1->1**   | 0.9                     | 0.75                       |

*Note: The mapping of inferred states (0, 1) to conceptual states (N(0,1)/N(5,1)) is based on the proximity of their learned means to the true means.*

#### Why the Learned Covariances May Be Inaccurate

Even though the learned means and transition probabilities are relatively close to the true values in this analytical example, the learned covariances can appear inaccurate. This is primarily due to the **very small dataset size** (only 10 data points) used for training. Estimating variances (and covariances) accurately requires a sufficient number of samples. With only a few data points assigned to each hidden state, the sample variance can deviate significantly from the true population variance, leading to less precise estimates for `covars_`.

### How to Choose the Number of Iterations (`n_iter`)

The `n_iter` parameter in `hmmlearn` specifies the **maximum number of iterations** for the Expectation-Maximization (EM) algorithm used to train Hidden Markov Models. The algorithm iteratively refines the HMM parameters (transition probabilities, emission means, and covariances) to maximize the likelihood of the observed data. It continues until either the change in log-likelihood between successive iterations falls below a certain threshold (convergence) or the maximum number of iterations (`n_iter`) is reached.

**Factors Influencing the Choice of `n_iter`:**

1.  **Convergence Criteria (Primary Indicator):** Ideally, you want the model to converge. `hmmlearn` models have a `monitor_` attribute (e.g., `model.monitor_.converged`) that indicates if convergence was achieved. If it consistently doesn't converge, `n_iter` might be too low.
2.  **Computational Cost:** Each iteration can be computationally expensive. Setting `n_iter` too high unnecessarily increases training time if the model converges earlier.
3.  **Model Complexity:** More complex HMMs (e.g., more hidden states, more mixture components in `GMMHMM`, or `full` covariance types) generally require more iterations to converge.
4.  **Data Characteristics:** Noisy data or data where the hidden states are not clearly separable might require more iterations.

**Practical Strategies for Choosing `n_iter`:**

1.  **Start with a Reasonable Default:** Values like `100` or `200` are common starting points.
2.  **Monitor Convergence:** During development, you can observe the log-likelihood progression. It should increase rapidly and then plateau.
3.  **Increase if Not Converging:** If the model reports non-convergence, incrementally increase `n_iter`.
4.  **Use Hyperparameter Optimization:** Tools like Optuna (as implemented in this project) can automatically search for an optimal `n_iter` range, balancing performance and computational cost. The objective function (maximizing log-likelihood) naturally guides Optuna to find an `n_iter` that allows for good performance without over-iterating.

In summary, the goal is to set `n_iter` high enough to allow convergence, but not excessively high. Hyperparameter optimization is the most robust way to determine an efficient `n_iter` for your specific problem.

## `hmmlearn.GaussianHMM`


This project uses `hmmlearn.GaussianHMM`, which is suitable for continuous observation data. For each hidden state, it assumes that the observed features follow a multivariate Gaussian distribution. The model learns:

*   **`n_components` (Number of Hidden States):** The number of distinct market regimes we assume exist.
*   **`startprob_` (Initial State Probabilities):** The probability of starting in each hidden state.
*   **`transmat_` (Transition Matrix):** A matrix where `transmat_[i, j]` is the probability of transitioning from hidden state `i` to hidden state `j`.
*   **`means_` (Mean Vectors):** For each hidden state, a vector representing the mean of the Gaussian distribution for the observed features.
*   **`covars_` (Covariance Matrices):** For each hidden state, a matrix representing the covariance of the Gaussian distribution for the observed features. The `covariance_type` parameter (`"full"`, `"diag"`, etc.) determines the structure of these matrices.

## `hmmlearn.GMMHMM`

In addition to `GaussianHMM`, this project also supports `hmmlearn.GMMHMM`. While `GaussianHMM` assumes a single Gaussian distribution for observations within each hidden state, `GMMHMM` allows for a **Gaussian Mixture Model (GMM)** to represent the emission probabilities for each state. This means that the observations within a single hidden state can be a mixture of several Gaussian distributions.

**When to use `GMMHMM`?**

`GMMHMM` is useful when the distribution of observations within a hidden state is more complex and cannot be adequately modeled by a single Gaussian distribution. For example, if a market regime exhibits bimodal or multimodal behavior in its features, `GMMHMM` can capture this complexity by using multiple mixture components (`n_mix`).

**Key parameters for `GMMHMM`:**

*   **`n_mix`:** This parameter specifies the number of Gaussian mixture components for each hidden state. A higher `n_mix` allows for more flexible modeling of the emission distributions but also increases model complexity and the risk of overfitting.

All other parameters like `n_components`, `covariance_type`, `n_iter`, and `random_state` function similarly to `GaussianHMM`.

## When to choose `HMMType.GAUSSIAN` vs. `HMMType.GMM`?

*   **`HMMType.GAUSSIAN` (Default):** This is the simpler and often sufficient choice. It assumes that the observations within each hidden state are well-approximated by a single multivariate Gaussian distribution. It's computationally less intensive and less prone to overfitting, especially with limited data. While conceptually `GaussianHMM` is a special case of `GMMHMM` with `n_mix=1`, `hmmlearn` provides a dedicated `GaussianHMM` class for efficiency and clarity.

*   **`HMMType.GMM`:** Choose `GMMHMM` when the distribution of observations within a hidden state is more complex and appears to be multimodal (i.e., it has multiple peaks or clusters). For example, if a "mean-reverting" regime sometimes exhibits small positive price changes and sometimes small negative price changes, but rarely zero, a single Gaussian might not capture this bimodal behavior well. `GMMHMM` can model such complex emission distributions by combining multiple Gaussian components.

## How to choose the number of mixture components (`n_mix`) for `HMMType.GMM`?

Choosing `n_mix` is similar to choosing the number of clusters in a Gaussian Mixture Model (GMM) for clustering. There's no single definitive method, but common approaches include:

1.  **Domain Knowledge:** If you have prior knowledge about the underlying data generation process and suspect specific multimodal behaviors within regimes, you might start with `n_mix` values that reflect these expectations.
2.  **Visual Inspection:** Plotting the distribution of features within identified regimes (after an initial HMM run with `GaussianHMM` or a small `n_mix`) can reveal multimodality, suggesting a need for a higher `n_mix`.
3.  **Information Criteria:** Use metrics like the Bayesian Information Criterion (BIC) or Akaike Information Criterion (AIC). Train `GMMHMM` models with varying `n_mix` values and choose the `n_mix` that minimizes BIC or AIC. These criteria penalize model complexity, helping to prevent overfitting.
4.  **Cross-Validation:** Split your data into training and validation sets. Train models with different `n_mix` values on the training set and evaluate their performance (e.g., log-likelihood) on the validation set. Choose the `n_mix` that yields the best performance.
5.  **Start Simple:** Begin with a small `n_mix` (e.g., 1 or 2) and gradually increase it if the model's performance is unsatisfactory or if visual inspection suggests more complex emission distributions. Be mindful of increased computational cost and the risk of overfitting with higher `n_mix` values.

It's generally recommended to start with `HMMType.GAUSSIAN` and only consider `HMMType.GMM` if the `GaussianHMM` proves insufficient for capturing the underlying data distributions within regimes.

## When to choose `HMMType.GAUSSIAN` vs. `HMMType.GMM`?

*   **`HMMType.GAUSSIAN` (Default):** This is the simpler and often sufficient choice. It assumes that the observations within each hidden state are well-approximated by a single multivariate Gaussian distribution. It's computationally less intensive and less prone to overfitting, especially with limited data. While conceptually `GaussianHMM` is a special case of `GMMHMM` with `n_mix=1`, `hmmlearn` provides a dedicated `GaussianHMM` class for efficiency and clarity.

*   **`HMMType.GMM`:** Choose `GMMHMM` when the distribution of observations within a hidden state is more complex and appears to be multimodal (i.e., it has multiple peaks or clusters). For example, if a "mean-reverting" regime sometimes exhibits small positive price changes and sometimes small negative price changes, but rarely zero, a single Gaussian might not capture this bimodal behavior well. `GMMHMM` can model such complex emission distributions by combining multiple Gaussian components.

## How to choose the number of mixture components (`n_mix`) for `HMMType.GMM`?

Choosing `n_mix` is similar to choosing the number of clusters in a Gaussian Mixture Model (GMM) for clustering. There's no single definitive method, but common approaches include:

1.  **Domain Knowledge:** If you have prior knowledge about the underlying data generation process and suspect specific multimodal behaviors within regimes, you might start with `n_mix` values that reflect these expectations.
2.  **Visual Inspection:** Plotting the distribution of features within identified regimes (after an initial HMM run with `GaussianHMM` or a small `n_mix`) can reveal multimodality, suggesting a need for a higher `n_mix`.
3.  **Information Criteria:** Use metrics like the Bayesian Information Criterion (BIC) or Akaike Information Criterion (AIC). Train `GMMHMM` models with varying `n_mix` values and choose the `n_mix` that minimizes BIC or AIC. These criteria penalize model complexity, helping to prevent overfitting.
4.  **Cross-Validation:** Split your data into training and validation sets. Train models with different `n_mix` values on the training set and evaluate their performance (e.g., log-likelihood) on the validation set. Choose the `n_mix` that yields the best performance.
5.  **Start Simple:** Begin with a small `n_mix` (e.g., 1 or 2) and gradually increase it if the model's performance is unsatisfactory or if visual inspection suggests more complex emission distributions. Be mindful of increased computational cost and the risk of overfitting with higher `n_mix` values.

It's generally recommended to start with `HMMType.GAUSSIAN` and only consider `HMMType.GMM` if the `GaussianHMM` proves insufficient for capturing the underlying data distributions within regimes.

## Understanding the Interpretation Column: `interpret_column`

The `interpret_column` parameter in `HMMRegimeIdentifier` is crucial for assigning meaningful labels to the hidden states discovered by the HMM. It's important to understand its role and how it relates to forward-looking data.

**Role of `interpret_column`:**

The HMM itself learns patterns in the `feature_labels` you provide. The `interpret_column` is *not* used during the HMM training or state inference process. Instead, after the HMM has identified the hidden states, the `interpret_regimes` method calculates the average value of the `interpret_column` within each of these states. This average helps in: 

*   **Labeling Regimes:** Assigning human-readable labels like "Mean-Reverting" or "Trend-Following (Up)" based on the typical behavior of this column within each state.
*   **Understanding State Characteristics:** Providing insight into what each hidden state represents in terms of the chosen interpretation variable.

**Should `interpret_column` be Forward-Looking?**

This is a critical design choice, especially in financial modeling:

*   **For Interpretation (Yes, often):** If your goal is to identify market regimes that *precede* certain future market behaviors, then setting `interpret_column` to a forward-looking variable (e.g., "next day's return", "future volatility") can be very powerful. For example, if a hidden state consistently shows a high average "next day's return", you can confidently label it as a "Bullish" regime. This helps you understand the predictive power of your identified regimes.

*   **For HMM Training (No, generally avoid):** It is crucial that the `feature_labels` (the actual inputs to the HMM) do *not* contain any information from the future that would not be available at the time of prediction. Including forward-looking data in `feature_labels` leads to **look-ahead bias** (or data leakage), which will make your model appear to perform exceptionally well during training but fail dramatically in real-world applications. The HMM's training and inference should only use historical or current information.

**In summary:**

*   You can and often should use a **forward-looking variable** for `interpret_column` to gain insights into the predictive nature of your identified regimes.
*   **Never include forward-looking variables in your `feature_labels`** unless you have a robust, non-leaky method for constructing them (e.g., features derived purely from past data that are predictive of future outcomes, but do not directly contain future values). The `HMMRegimeIdentifier` is designed to prevent this by separating the `interpret_column` from the `feature_labels` used for HMM training.

## Dealing with Irregularly Sampled Data

While standard HMMs in `hmmlearn` do not explicitly model time intervals between observations, this implementation addresses irregularly sampled data by **including `time_since_last_obs` as one of the input features**.

By incorporating `time_since_last_obs` into the feature set, the HMM's emission probabilities for each hidden state will learn the typical distribution of time intervals associated with that state, alongside other features like `prev_day_price_delta` and `volatility`. For example:

*   A "mean-reverting" regime might be characterized by smaller `prev_day_price_delta` values, lower `volatility`, and potentially more frequent (smaller `time_since_last_obs`) observations if the market is actively reacting to a mean.
*   A "trend-following" regime might show consistent `prev_day_price_delta` in one direction, higher `volatility`, and potentially different patterns in `time_since_last_obs` depending on how the trend manifests in observation frequency.

This approach allows the HMM to implicitly leverage the information contained in the irregular sampling intervals to better distinguish between different underlying market regimes.

## Feature Standardization

The `HMMRegimeIdentifier` class includes an option to standardize the features before training the HMM. This is controlled by the `standardize_features` parameter in the constructor (default is `True`).

**Why Standardize?**

The `GaussianHMM` assumes that the features for each hidden state are drawn from a multivariate Gaussian distribution. The scale and range of these features can significantly impact the model's performance:

*   **Equal Weighting:** Features with larger scales (e.g., `rsi_proxy` ranging from 0-100) can dominate features with smaller scales (e.g., `price_delta` ranging from -2 to 2). Standardization rescales all features to have a mean of 0 and a standard deviation of 1, ensuring that the HMM gives them more equal consideration when calculating distances and probabilities.
*   **Improved Convergence:** Standardization can help the Expectation-Maximization (EM) algorithm used for training converge faster and more reliably.

When `standardize_features=True`, the class uses `sklearn.preprocessing.StandardScaler` to fit and transform the feature data. The learned scaler is stored and used for any subsequent data transformations, ensuring consistency.

## Scalability Considerations

Understanding how HMM methods scale with data size and complexity is crucial for real-world applications. The computational complexity of `hmmlearn`'s `GaussianHMM` and `GMMHMM` depends on several key variables:

*   `T`: Number of data points (length of the observation sequence).
*   `M`: Number of features (dimensions of each observation).
*   `N`: Number of hidden states (`n_components`).
*   `K`: Number of mixture components per state (`n_mix`, only for `GMMHMM`).

### Computational Complexity:

1.  **Training (Baum-Welch Algorithm):**
    This is typically the most computationally intensive part, as it's an iterative expectation-maximization (EM) algorithm. Each iteration involves forward and backward passes.
    *   **`GaussianHMM`**: Approximately **O(num_iterations * T * N² * M)**
    *   **`GMMHMM`**: Approximately **O(num_iterations * T * N² * M * K)**

2.  **Inference (Viterbi Algorithm) and Scoring (Forward Algorithm):**
    These algorithms are used to find the most likely sequence of hidden states or to calculate the likelihood of an observation sequence given the model.
    *   **`GaussianHMM`**: Approximately **O(T * N² * M)**
    *   **`GMMHMM`**: Approximately **O(T * N² * M * K)**

### How it Scales:

*   **Number of Data Points (`T`):**
    *   **Linear Scaling:** All core HMM algorithms (training, inference, scoring) scale **linearly** with the number of data points. This is generally good. If you double your data points, the computation time will roughly double. This makes HMMs reasonably scalable for very long time series.

*   **Number of Features (`M`):**
    *   **Linear Scaling:** All core HMM algorithms scale **linear** with the number of features. This is also quite favorable. Doubling the number of features will roughly double the computation time.

*   **Number of Hidden States (`N`):}
    *   **Quadratic Scaling:** This is the most significant bottleneck. The algorithms scale **quadratically** with the number of hidden states (`N²`). This means that if you double the number of hidden states, the computation time will increase by a factor of four. This is why choosing a relatively small number of hidden states is often practical.

*   **Number of Mixture Components (`K`) - for `GMMHMM` only:**
    *   **Linear Scaling:** The algorithms scale **linear** with the number of mixture components. This is manageable. Increasing `n_mix` will increase computation, but not as dramatically as increasing `n_components`.

### Practical Considerations:

*   **Memory Usage:**
    *   For `full` covariance types, the covariance matrices grow quadratically with the number of features (`M²`). If you have many features and use `full` covariance, memory can become a significant constraint. `diag` or `spherical` covariance types are more memory-efficient.
    *   Storing the observation sequence (`T * M`) can also consume substantial memory for very large datasets.

*   **Convergence:**
    *   The Baum-Welch Algorithm (for training) is an iterative algorithm that can sometimes be slow to converge, especially for complex models (many states, many mixtures, high-dimensional features) or if initialized poorly. It can also get stuck in local optima.

*   **Data Sparsity:**
    *   If your data is very sparse, especially with many features, it can make parameter estimation challenging and potentially lead to unstable models.

In summary, HMMs generally scale well with the number of data points and features. The primary factor limiting scalability is often the **number of hidden states**, due to its quadratic impact on computation. When using `GMMHMM`, the number of mixture components also adds a linear factor to the complexity.

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

Once the installation is complete, you can run the examples provided in the `examples` folder.

### Running the Synthetic Data Example

To run the main script with synthetic data:

```bash
.\venv\Scripts\python .\examples\run_synthetic_example.py
```

This will:
1.  Generate synthetic data.
2.  Train the HMM.
3.  Infer the hidden market regimes.
4.  Print the learned HMM parameters and inferred regime labels.
5.  Display two plots visualizing the results.

You can modify the `if __name__ == "__main__":` block in `run_synthetic_example.py` to:
*   Adjust the number of synthetic samples (`n_samples`).
*   Change the HMM parameters (`n_components`, `covariance_type`, `n_iter`).
*   Adapt the `interpret_column` and `feature_cols` to your specific data if you replace the synthetic data generation with your own dataset.

### Running the Analytical Example

To run the simple analytical example:

```bash
.\venv\Scripts\python .\examples\run_analytical_example.py
```

This will demonstrate the HMM's ability to learn parameters from a simple dataset with known underlying states.

### Running the Dishonest Casino Example

To run the simulated dishonest casino example:

```bash
.\venv\Scripts\python .\examples\run_casino_example.py
```

This script simulates the "Dishonest Casino" problem described in the technical section and uses an HMM to infer which die (fair or loaded) was used at each step. It will print the learned HMM parameters and visualize the true versus inferred hidden states.

### Hyperparameter Optimization with Optuna

To run the hyperparameter optimization example using Optuna:

```bash
.\venv\Scripts\python .\examples\run_optimization_example.py
```

This script now directly initializes the `HMMRegimeIdentifier` with `run_optimization=True`, triggering the Optuna optimization process internally. It will print the best trial's value (log-likelihood) and its corresponding parameters found during the optimization.

## Project Structure

```
.
├── hmm_regime_identification.py
├── README.md
├── requirements.txt
├── examples/
│   ├── run_analytical_example.py
│   ├── run_synthetic_example.py
│   ├── run_optimization_example.py
│   └── run_casino_example.py
└── venv/
```