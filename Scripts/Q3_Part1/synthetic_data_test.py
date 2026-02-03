"""
Synthetic Data Tests for Deep Context-Dependent Choice Model
Replicating the experiments from Zhang et al. (2025)
"""

import numpy as np
import tensorflow as tf
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from deep_context_choice_model import (
    DeepContextDependentChoiceModel,
    DeepContextChoiceModelTrainer
)


class SyntheticDataGenerator:
    """
    Generate synthetic data for testing the deep context-dependent choice model.
    """

    def __init__(
        self,
        context_dim: int = 10,
        product_dim: int = 5,
        n_products: int = 5,
        context_type: str = 'nonlinear',
        noise_level: float = 0.1,
        seed: int = 42
    ):
        """
        Initialize synthetic data generator.

        Args:
            context_dim: Dimension of context features
            product_dim: Dimension of product features
            n_products: Number of products in choice set
            context_type: Type of context dependency ('linear', 'nonlinear', 'complex')
            noise_level: Noise level in utility
            seed: Random seed
        """
        self.context_dim = context_dim
        self.product_dim = product_dim
        self.n_products = n_products
        self.context_type = context_type
        self.noise_level = noise_level
        self.seed = seed

        np.random.seed(seed)
        tf.random.set_seed(seed)

        # Generate true parameters
        self._generate_true_parameters()

    def _generate_true_parameters(self):
        """Generate true parameters for data generation."""
        # Context weights
        self.context_weights = np.random.randn(self.context_dim, self.product_dim)

        # Product-specific parameters
        self.product_bias = np.random.randn(self.n_products) * 0.5

        # Interaction parameters (for nonlinear context dependency)
        if self.context_type in ['nonlinear', 'complex']:
            self.interaction_weights = np.random.randn(
                self.context_dim,
                self.product_dim,
                self.n_products
            ) * 0.3

    def _compute_true_utility(
        self,
        context: np.ndarray,
        product_features: np.ndarray
    ) -> np.ndarray:
        """
        Compute true utility based on context and product features.

        Args:
            context: Context features [batch_size, context_dim]
            product_features: Product features [batch_size, n_products, product_dim]

        Returns:
            Utilities [batch_size, n_products]
        """
        batch_size = context.shape[0]
        utilities = np.zeros((batch_size, self.n_products))

        for i in range(batch_size):
            ctx = context[i]

            for j in range(self.n_products):
                prod_feat = product_features[i, j]

                # Base utility: linear combination
                base_utility = self.product_bias[j]
                base_utility += np.sum(prod_feat * np.dot(ctx, self.context_weights))

                # Context-dependent part
                if self.context_type == 'linear':
                    # Simple linear context dependency
                    utilities[i, j] = base_utility

                elif self.context_type == 'nonlinear':
                    # Nonlinear context dependency
                    interaction = np.sum(
                        ctx[:, None] * prod_feat[None, :] * self.interaction_weights[:, :, j]
                    )
                    # Add squared terms for nonlinearity
                    nonlinear_term = 0.1 * np.sum(ctx**2 * prod_feat.sum())
                    utilities[i, j] = base_utility + interaction + nonlinear_term

                elif self.context_type == 'complex':
                    # Complex context dependency with trigonometric functions
                    interaction = np.sum(
                        ctx[:, None] * prod_feat[None, :] * self.interaction_weights[:, :, j]
                    )
                    complex_term = 0.1 * np.sum(np.sin(ctx) * np.cos(prod_feat.sum()))
                    utilities[i, j] = base_utility + interaction + complex_term

        # Add Gumbel noise for stochasticity
        gumbel_noise = np.random.gumbel(0, self.noise_level, utilities.shape)
        utilities += gumbel_noise

        return utilities

    def generate_data(
        self,
        n_samples: int,
        return_true_utilities: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Generate synthetic choice data.

        Args:
            n_samples: Number of samples to generate
            return_true_utilities: Whether to return true utilities

        Returns:
            Dictionary containing:
                - context: Context features
                - product_features: Product features
                - choices: One-hot encoded choices
                - true_utilities: True utilities (if requested)
        """
        # Generate random context
        context = np.random.randn(n_samples, self.context_dim)

        # Generate random product features
        product_features = np.random.randn(n_samples, self.n_products, self.product_dim)

        # Compute true utilities
        true_utilities = self._compute_true_utility(context, product_features)

        # Generate choices based on MNL
        exp_utilities = np.exp(true_utilities)
        probabilities = exp_utilities / exp_utilities.sum(axis=1, keepdims=True)

        # Sample choices
        choices = np.zeros((n_samples, self.n_products))
        for i in range(n_samples):
            chosen = np.random.choice(self.n_products, p=probabilities[i])
            choices[i, chosen] = 1

        data = {
            'context': context.astype(np.float32),
            'product_features': product_features.astype(np.float32),
            'choices': choices.astype(np.float32),
        }

        if return_true_utilities:
            data['true_utilities'] = true_utilities.astype(np.float32)
            data['true_probabilities'] = probabilities.astype(np.float32)

        return data

    def save_to_csv(self, data: Dict[str, np.ndarray], filename: str):
        """Save generated data to CSV."""
        import pandas as pd
        
        print(f"Saving data to {filename}...")
        
        # Vectorized implementation
        N = data['context'].shape[0]
        
        # 1. Context df
        context_df = pd.DataFrame(
            data['context'], 
            columns=[f'context_{i}' for i in range(data['context'].shape[1])]
        )
        
        # 2. Product features df
        # Flatten: (N, n_prods, n_feats) -> (N, n_prods * n_feats)
        n_prods = data['product_features'].shape[1]
        n_feats = data['product_features'].shape[2]
        prod_flat = data['product_features'].reshape(N, -1)
        prod_cols = [f'prod_{p}_feat_{f}' for p in range(n_prods) for f in range(n_feats)]
        prod_df = pd.DataFrame(prod_flat, columns=prod_cols)
        
        # 3. Choices df
        # Convert one-hot to index
        choice_idx = np.argmax(data['choices'], axis=1)
        choice_df = pd.DataFrame(choice_idx, columns=['choice'])
        
        # Combine
        full_df = pd.concat([context_df, prod_df, choice_df], axis=1)
        
        # Add true utilities if present
        if 'true_utilities' in data:
            util_cols = [f'true_utility_{p}' for p in range(n_prods)]
            util_df = pd.DataFrame(data['true_utilities'], columns=util_cols)
            full_df = pd.concat([full_df, util_df], axis=1)
            
        full_df.to_csv(filename, index=False)
        print(f"Successfully saved {N} samples to {filename}")


def test_synthetic_data_linear():
    """
    Test 1: Linear context dependency.
    The model should easily learn this.
    """
    print("\n" + "="*80)
    print("Test 1: Linear Context Dependency")
    print("="*80)

    # Generate data
    data_gen = SyntheticDataGenerator(
        context_dim=10,
        product_dim=5,
        n_products=5,
        context_type='linear',
        noise_level=0.1,
        seed=42
    )

    train_data = data_gen.generate_data(5000, return_true_utilities=True)
    val_data = data_gen.generate_data(1000, return_true_utilities=True)
    test_data = data_gen.generate_data(1000, return_true_utilities=True)

    # Save to CSV
    data_gen.save_to_csv(train_data, 'synthetic_data_linear_train.csv')
    data_gen.save_to_csv(test_data, 'synthetic_data_linear_test.csv')

    # Create model
    model = DeepContextDependentChoiceModel(
        context_dim=10,
        product_dim=5,
        context_hidden_dims=[64, 32],
        product_hidden_dims=[32, 16],
        context_latent_dim=8,
        product_latent_dim=8,
        interaction_dims=[32, 16],
        use_attention=False,  # Simple case
        temperature=1.0
    )

    # Create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((
        train_data['context'],
        train_data['product_features'],
        train_data['choices']
    )).batch(64).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((
        val_data['context'],
        val_data['product_features'],
        val_data['choices']
    )).batch(64).prefetch(tf.data.AUTOTUNE)

    # Train model
    # CRITICAL FIX: Increased learning rate from 1e-3 to 1e-2 (10x) to prevent gradient death
    trainer = DeepContextChoiceModelTrainer(
        model=model,
        learning_rate=1e-2,
        optimizer='adam',
        l2_regularization=1e-4
    )

    history = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=50,
        verbose=1
    )

    # Evaluate on test set
    print("\nTest Set Evaluation:")
    test_outputs = model.predict_choice(
        tf.constant(test_data['context']),
        tf.constant(test_data['product_features'])
    )
    test_probs = test_outputs.numpy()

    # Compute metrics
    predicted_choices = np.argmax(test_probs, axis=1)
    true_choices = np.argmax(test_data['choices'], axis=1)
    accuracy = np.mean(predicted_choices == true_choices)

    # NLL
    epsilon = 1e-10
    nll = -np.mean(np.sum(test_data['choices'] * np.log(test_probs + epsilon), axis=1))

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test NLL: {nll:.4f}")

    # Compare with true probabilities
    true_probs = test_data['true_probabilities']
    prob_mse = np.mean((test_probs - true_probs)**2)
    print(f"Probability MSE (vs true): {prob_mse:.4f}")

    return history, model, test_data


def test_synthetic_data_nonlinear():
    """
    Test 2: Nonlinear context dependency.
    Tests the model's ability to capture nonlinear interactions.
    """
    print("\n" + "="*80)
    print("Test 2: Nonlinear Context Dependency")
    print("="*80)

    # Generate data
    data_gen = SyntheticDataGenerator(
        context_dim=10,
        product_dim=5,
        n_products=5,
        context_type='nonlinear',
        noise_level=0.15,
        seed=42
    )

    train_data = data_gen.generate_data(10000, return_true_utilities=True)
    val_data = data_gen.generate_data(2000, return_true_utilities=True)
    test_data = data_gen.generate_data(2000, return_true_utilities=True)

    # Create model with attention
    model = DeepContextDependentChoiceModel(
        context_dim=10,
        product_dim=5,
        context_hidden_dims=[128, 64, 32],
        product_hidden_dims=[64, 32],
        context_latent_dim=16,
        product_latent_dim=16,
        interaction_dims=[64, 32, 16],
        use_attention=True,
        temperature=1.0
    )

    # Create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((
        train_data['context'],
        train_data['product_features'],
        train_data['choices']
    )).shuffle(10000).batch(128).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((
        val_data['context'],
        val_data['product_features'],
        val_data['choices']
    )).batch(128).prefetch(tf.data.AUTOTUNE)

    # Train model with learning rate schedule
    # CRITICAL FIX: Start high at 1e-2, then decay to allow convergence
    import tensorflow.keras as keras

    # Learning rate schedule: Start high, decay after model "wakes up"
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=500,  # Decay every 500 steps
        decay_rate=0.95,   # Multiply by 0.95
        staircase=True
    )

    trainer = DeepContextChoiceModelTrainer(
        model=model,
        learning_rate=lr_schedule,
        optimizer='adam',
        l2_regularization=1e-4
    )

    # Train for much longer to ensure convergence
    # With ResNet skip connection, accuracy MUST reach >= linear model (66%)
    history = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=500,  # Increased from 100 to ensure convergence
        verbose=1
    )

    # Evaluate on test set
    print("\nTest Set Evaluation:")
    test_outputs = model.predict_choice(
        tf.constant(test_data['context']),
        tf.constant(test_data['product_features'])
    )
    test_probs = test_outputs.numpy()

    # Compute metrics
    predicted_choices = np.argmax(test_probs, axis=1)
    true_choices = np.argmax(test_data['choices'], axis=1)
    accuracy = np.mean(predicted_choices == true_choices)

    epsilon = 1e-10
    nll = -np.mean(np.sum(test_data['choices'] * np.log(test_probs + epsilon), axis=1))

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test NLL: {nll:.4f}")

    # Compare with true probabilities
    true_probs = test_data['true_probabilities']
    prob_mse = np.mean((test_probs - true_probs)**2)
    print(f"Probability MSE (vs true): {prob_mse:.4f}")

    return history, model, test_data


def test_varying_choice_set_sizes():
    """
    Test 3: Varying choice set sizes.
    Tests model's ability to handle different assortment sizes.
    """
    print("\n" + "="*80)
    print("Test 3: Varying Choice Set Sizes")
    print("="*80)

    results = {}

    for n_products in [3, 5, 10, 20]:
        print(f"\nTesting with {n_products} products...")

        # Generate data
        data_gen = SyntheticDataGenerator(
            context_dim=10,
            product_dim=5,
            n_products=n_products,
            context_type='nonlinear',
            noise_level=0.1,
            seed=42
        )

        train_data = data_gen.generate_data(5000)
        test_data = data_gen.generate_data(1000)

        # Create model
        model = DeepContextDependentChoiceModel(
            context_dim=10,
            product_dim=5,
            context_hidden_dims=[64, 32],
            product_hidden_dims=[32, 16],
            context_latent_dim=12,
            product_latent_dim=12,
            interaction_dims=[48, 24],
            use_attention=True
        )

        # Create datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((
            train_data['context'],
            train_data['product_features'],
            train_data['choices']
        )).shuffle(5000).batch(64).prefetch(tf.data.AUTOTUNE)

        # Train model
        trainer = DeepContextChoiceModelTrainer(
            model=model,
            learning_rate=1e-3,
            optimizer='adam',
            l2_regularization=1e-4
        )

        history = trainer.train(
            train_dataset=train_dataset,
            val_dataset=None,
            epochs=30,
            verbose=0
        )

        # Evaluate
        test_probs = model.predict_choice(
            tf.constant(test_data['context']),
            tf.constant(test_data['product_features'])
        ).numpy()

        predicted_choices = np.argmax(test_probs, axis=1)
        true_choices = np.argmax(test_data['choices'], axis=1)
        accuracy = np.mean(predicted_choices == true_choices)

        epsilon = 1e-10
        nll = -np.mean(np.sum(test_data['choices'] * np.log(test_probs + epsilon), axis=1))

        print(f"  Accuracy: {accuracy:.4f}, NLL: {nll:.4f}")

        results[n_products] = {
            'accuracy': accuracy,
            'nll': nll,
            'history': history
        }

    return results


def test_substitution_patterns():
    """
    Test 4: Substitution patterns.
    Tests IIA (Independence of Irrelevant Alternatives) violations.
    """
    print("\n" + "="*80)
    print("Test 4: Substitution Patterns and IIA Violations")
    print("="*80)

    # Generate data
    data_gen = SyntheticDataGenerator(
        context_dim=10,
        product_dim=5,
        n_products=5,
        context_type='complex',
        noise_level=0.1,
        seed=42
    )

    # Generate test contexts
    n_test = 1000
    test_context = np.random.randn(n_test, 10).astype(np.float32)

    # Create two scenarios: full choice set and reduced choice set
    full_products = np.random.randn(n_test, 5, 5).astype(np.float32)

    # Train model on full data
    train_data = data_gen.generate_data(10000)

    model = DeepContextDependentChoiceModel(
        context_dim=10,
        product_dim=5,
        context_hidden_dims=[128, 64, 32],
        product_hidden_dims=[64, 32],
        context_latent_dim=16,
        product_latent_dim=16,
        interaction_dims=[64, 32, 16],
        use_attention=True
    )

    train_dataset = tf.data.Dataset.from_tensor_slices((
        train_data['context'],
        train_data['product_features'],
        train_data['choices']
    )).shuffle(10000).batch(128).prefetch(tf.data.AUTOTUNE)

    trainer = DeepContextChoiceModelTrainer(
        model=model,
        learning_rate=1e-3,
        optimizer='adam',
        l2_regularization=1e-4
    )

    trainer.train(train_dataset, epochs=50, verbose=0)

    # Test with full choice set
    probs_full = model.predict_choice(
        tf.constant(test_context),
        tf.constant(full_products)
    ).numpy()

    # Test with reduced choice set (remove product 2)
    available_mask = np.ones((n_test, 5), dtype=np.float32)
    available_mask[:, 2] = 0

    probs_reduced = model.predict_choice(
        tf.constant(test_context),
        tf.constant(full_products),
        tf.constant(available_mask)
    ).numpy()

    # Check IIA: ratio of probabilities should change
    # Under MNL, P(i)/P(j) should be constant, but with context-dependence it changes
    print("\nChecking substitution patterns:")
    print("Product 0 vs Product 1 probability ratio:")
    ratio_full = probs_full[:10, 0] / (probs_full[:10, 1] + 1e-10)
    ratio_reduced = probs_reduced[:10, 0] / (probs_reduced[:10, 1] + 1e-10)

    print("Full choice set:", ratio_full)
    print("Reduced choice set:", ratio_reduced)
    print("Relative change:", np.abs(ratio_full - ratio_reduced) / ratio_full)

    # Average ratio change
    avg_ratio_change = np.mean(np.abs(ratio_full - ratio_reduced) / (ratio_full + 1e-10))
    print(f"\nAverage ratio change: {avg_ratio_change:.4f}")
    print("(Significant change indicates IIA violation, which is realistic)")

    return {
        'probs_full': probs_full,
        'probs_reduced': probs_reduced,
        'ratio_change': avg_ratio_change
    }


def plot_training_curves(history: Dict[str, List[float]], save_path: str = None):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(history['train_loss'], label='Train')
    if history['val_loss']:
        axes[0].plot(history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Negative Log-Likelihood')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history['train_accuracy'], label='Train')
    if history['val_accuracy']:
        axes[1].plot(history['val_accuracy'], label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def run_all_tests():
    """Run all synthetic data tests."""
    print("\n" + "="*80)
    print("DEEP CONTEXT-DEPENDENT CHOICE MODEL - SYNTHETIC DATA TESTS")
    print("="*80)

    results = {}

    # Test 1: Linear
    history1, model1, data1 = test_synthetic_data_linear()
    results['linear'] = {'history': history1, 'model': model1, 'data': data1}

    # Test 2: Nonlinear
    history2, model2, data2 = test_synthetic_data_nonlinear()
    results['nonlinear'] = {'history': history2, 'model': model2, 'data': data2}

    # Test 3: Varying sizes
    results['varying_sizes'] = test_varying_choice_set_sizes()

    # Test 4: Substitution patterns
    results['substitution'] = test_substitution_patterns()

    print("\n" + "="*80)
    print("ALL TESTS COMPLETED")
    print("="*80)

    return results


if __name__ == '__main__':
    # Set random seeds
    np.random.seed(42)
    tf.random.set_seed(42)

    # Run all tests
    results = run_all_tests()

    # Plot some results
    print("\nGenerating plots...")
    plot_training_curves(results['linear']['history'], 'linear_training.png')
    plot_training_curves(results['nonlinear']['history'], 'nonlinear_training.png')

    print("\nDone!")
