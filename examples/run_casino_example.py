import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmm_regime_identification import HMMRegimeIdentifier


# Function to simulate the dishonest casino
def simulate_casino(num_rolls=10000, fair_prob=1 / 6, loaded_prob_6=0.5, switch_prob=0.05, random_state=42):
    np.random.seed(random_state)

    # States: 0 for Fair Die, 1 for Loaded Die
    # Emission probabilities (simplified for GaussianHMM by treating rolls as continuous)
    # Fair die: mean ~3.5, small variance
    # Loaded die: mean shifted towards 6, larger variance

    # True hidden states
    true_states = []
    # Observed rolls
    observations = []

    current_state = np.random.choice([0, 1])  # Start with either fair or loaded

    for _ in range(num_rolls):
        true_states.append(current_state)

        if current_state == 0:  # Fair Die
            roll = np.random.randint(1, 7)  # Simulate a fair die roll
        else:  # Loaded Die
            # Simulate a loaded die (e.g., 50% chance of 6, rest distributed among 1-5)
            if np.random.rand() < loaded_prob_6:
                roll = 6
            else:
                roll = np.random.randint(1, 6)  # Rolls 1-5
        observations.append(roll)

        # Decide whether to switch state
        if np.random.rand() < switch_prob:
            current_state = 1 - current_state  # Flip state

    return np.array(observations).reshape(-1, 1), np.array(true_states)


if __name__ == "__main__":
    print("Simulating Dishonest Casino Data...")
    observations, true_states = simulate_casino(num_rolls=10000)
    print(f"Generated {len(observations)} observations.")

    # Convert to DataFrame for HMMRegimeIdentifier
    df_casino = pd.DataFrame(observations, columns=["roll"])

    # Define interpret_column and feature labels
    interpret_col = "roll"
    feature_cols = ["roll"]

    # Initialize HMMRegimeIdentifier
    regime_identifier = HMMRegimeIdentifier(
        df=df_casino,
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

    # Map inferred states to true states for better comparison
    # This is a heuristic mapping based on the means of the observed rolls
    # The state with a mean closer to 3.5 (fair die) is mapped to 0, and the other to 1.
    inferred_states = regime_identifier.df["regime"].values

    # Get the learned means from the HMM model
    learned_means = regime_identifier.model.means_

    if abs(learned_means[0, 0] - 3.5) < abs(learned_means[1, 0] - 3.5):
        mapping = {0: 0, 1: 1}  # Inferred state 0 is Fair, 1 is Loaded
    else:
        mapping = {0: 1, 1: 0}  # Inferred state 0 is Loaded, 1 is Fair

    mapped_inferred_states = np.array([mapping[s] for s in inferred_states])

    # Evaluate accuracy
    accuracy = np.mean(mapped_inferred_states == true_states)
    print(f"\nAccuracy of inferred states: {accuracy:.2f}")

    # Visualize results
    plt.figure(figsize=(15, 7))
    plt.subplot(2, 1, 1)
    plt.plot(observations, label="Observed Rolls", alpha=0.7)
    plt.scatter(range(len(observations)), observations, c=true_states, cmap="coolwarm", s=20, label="True States")
    plt.title("Observed Rolls with True Hidden States")
    plt.xlabel("Roll Number")
    plt.ylabel("Roll Value")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(mapped_inferred_states, drawstyle="steps-post", label="Inferred States", alpha=0.7)
    plt.plot(true_states, drawstyle="steps-post", label="True States", linestyle="--", alpha=0.7)
    plt.title("True vs. Inferred Hidden States")
    plt.xlabel("Roll Number")
    plt.ylabel("State (0=Fair, 1=Loaded)")
    plt.yticks([0, 1])
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
