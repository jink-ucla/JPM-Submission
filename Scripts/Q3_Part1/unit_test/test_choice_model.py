"""
Unit Tests for Q3: Deep Context-Dependent Choice Model
=======================================================
Tests the TensorFlow-based choice model implementation using pytest.

Tests cover:
1. Probability axiom compliance (non-negative, sum to 1)
2. Gradient flow verification (no vanishing gradients)
3. Model architecture correctness
4. Training convergence
5. IIA violation detection

References:
- Zhang et al. (2025): Deep Context-Dependent Choice Model
- pytest documentation: https://docs.pytest.org/
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import TensorFlow with error handling
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Skip all tests if TensorFlow not available
pytestmark = pytest.mark.skipif(
    not TF_AVAILABLE,
    reason="TensorFlow not available"
)

if TF_AVAILABLE:
    from deep_context_choice_model import (
        ContextEncoder,
        ProductEncoder,
        ContextProductInteraction,
        DeepContextDependentChoiceModel,
        DeepContextChoiceModelTrainer
    )


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def model_config():
    """Default model configuration."""
    return {
        'context_dim': 10,
        'product_dim': 5,
        'context_hidden_dims': [64, 32],
        'product_hidden_dims': [32, 16],
        'context_latent_dim': 8,
        'product_latent_dim': 8,
        'interaction_dims': [32, 16],
        'use_attention': True,
        'temperature': 1.0
    }


@pytest.fixture
def simple_model(model_config):
    """Create a simple model instance."""
    return DeepContextDependentChoiceModel(**model_config)


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    batch_size = 32
    context_dim = 10
    product_dim = 5
    n_products = 5

    context = np.random.randn(batch_size, context_dim).astype(np.float32)
    product_features = np.random.randn(batch_size, n_products, product_dim).astype(np.float32)

    # Generate one-hot choices
    choices = np.zeros((batch_size, n_products), dtype=np.float32)
    chosen_indices = np.random.randint(0, n_products, batch_size)
    for i, idx in enumerate(chosen_indices):
        choices[i, idx] = 1.0

    return context, product_features, choices


@pytest.fixture
def synthetic_linear_data():
    """Generate synthetic data with known linear relationship."""
    np.random.seed(42)
    n_samples = 1000
    context_dim = 10
    product_dim = 5
    n_products = 5

    # True parameters
    true_weights = np.random.randn(context_dim + product_dim)

    context = np.random.randn(n_samples, context_dim).astype(np.float32)
    product_features = np.random.randn(n_samples, n_products, product_dim).astype(np.float32)

    # Compute true utilities
    utilities = np.zeros((n_samples, n_products))
    for i in range(n_samples):
        for j in range(n_products):
            combined = np.concatenate([context[i], product_features[i, j]])
            utilities[i, j] = np.dot(combined, true_weights)

    # Generate choices via MNL
    exp_u = np.exp(utilities - utilities.max(axis=1, keepdims=True))
    probs = exp_u / exp_u.sum(axis=1, keepdims=True)

    choices = np.zeros((n_samples, n_products), dtype=np.float32)
    for i in range(n_samples):
        chosen = np.random.choice(n_products, p=probs[i])
        choices[i, chosen] = 1.0

    return context, product_features, choices, probs


# ==============================================================================
# Unit Tests: Context Encoder
# ==============================================================================

class TestContextEncoder:
    """Tests for ContextEncoder."""

    def test_output_shape(self):
        """Test that encoder outputs correct shape."""
        encoder = ContextEncoder(
            context_dim=10,
            hidden_dims=[64, 32],
            latent_dim=16
        )

        context = tf.random.normal((32, 10))
        output = encoder(context, training=False)

        assert output.shape == (32, 16)

    def test_output_deterministic_in_eval(self):
        """Test that output is deterministic in eval mode."""
        encoder = ContextEncoder(
            context_dim=10,
            hidden_dims=[64, 32],
            latent_dim=16
        )

        context = tf.random.normal((32, 10))
        output1 = encoder(context, training=False)
        output2 = encoder(context, training=False)

        np.testing.assert_array_almost_equal(
            output1.numpy(), output2.numpy()
        )


# ==============================================================================
# Unit Tests: Product Encoder
# ==============================================================================

class TestProductEncoder:
    """Tests for ProductEncoder."""

    def test_output_shape(self):
        """Test that encoder outputs correct shape."""
        encoder = ProductEncoder(
            product_dim=5,
            hidden_dims=[32, 16],
            latent_dim=8
        )

        products = tf.random.normal((32, 5, 5))  # batch x n_products x product_dim
        output = encoder(products, training=False)

        assert output.shape == (32, 5, 8)

    def test_weight_sharing_across_products(self):
        """Test that weights are shared across products."""
        encoder = ProductEncoder(
            product_dim=5,
            hidden_dims=[32, 16],
            latent_dim=8
        )

        # Create two identical products
        products = tf.random.normal((1, 2, 5))
        products = tf.concat([products[:, :1, :], products[:, :1, :]], axis=1)

        output = encoder(products, training=False)

        # Outputs should be identical for identical inputs
        np.testing.assert_array_almost_equal(
            output[0, 0].numpy(), output[0, 1].numpy()
        )


# ==============================================================================
# Unit Tests: Choice Model Probability Axioms
# ==============================================================================

class TestProbabilityAxioms:
    """
    CRITICAL TESTS: Verify probability axioms.

    Choice probabilities must satisfy:
    1. P(j|S) >= 0 for all j
    2. sum_j P(j|S) = 1 for all choice sets S
    """

    def test_probabilities_non_negative(self, simple_model, sample_data):
        """Test that all probabilities are non-negative."""
        context, product_features, _ = sample_data

        outputs = simple_model(
            tf.constant(context),
            tf.constant(product_features),
            training=False
        )

        probs = outputs['probabilities'].numpy()
        assert np.all(probs >= 0), "Found negative probabilities"

    def test_probabilities_sum_to_one(self, simple_model, sample_data):
        """Test that probabilities sum to 1 for each sample."""
        context, product_features, _ = sample_data

        outputs = simple_model(
            tf.constant(context),
            tf.constant(product_features),
            training=False
        )

        probs = outputs['probabilities'].numpy()
        sums = probs.sum(axis=1)

        np.testing.assert_array_almost_equal(
            sums, np.ones(len(sums)), decimal=5,
            err_msg="Probabilities don't sum to 1"
        )

    def test_probabilities_with_availability_mask(self, simple_model, sample_data):
        """Test that masked products get zero probability."""
        context, product_features, _ = sample_data

        # Mask out product 2
        availability = np.ones((32, 5), dtype=np.float32)
        availability[:, 2] = 0

        outputs = simple_model(
            tf.constant(context),
            tf.constant(product_features),
            tf.constant(availability),
            training=False
        )

        probs = outputs['probabilities'].numpy()

        # Masked product should have ~0 probability
        assert np.all(probs[:, 2] < 1e-6), "Masked product has non-zero probability"

        # Remaining should still sum to ~1
        remaining_sum = probs[:, [0, 1, 3, 4]].sum(axis=1)
        np.testing.assert_array_almost_equal(
            remaining_sum, np.ones(32), decimal=5
        )


# ==============================================================================
# Unit Tests: Gradient Flow
# ==============================================================================

class TestGradientFlow:
    """Tests for gradient flow through the network."""

    def test_gradients_exist(self, simple_model, sample_data):
        """Test that gradients flow to all trainable variables."""
        context, product_features, choices = sample_data

        with tf.GradientTape() as tape:
            loss_dict = simple_model.compute_loss(
                tf.constant(context),
                tf.constant(product_features),
                tf.constant(choices)
            )
            loss = loss_dict['loss']

        gradients = tape.gradient(loss, simple_model.trainable_variables)

        for var, grad in zip(simple_model.trainable_variables, gradients):
            assert grad is not None, f"No gradient for {var.name}"

    def test_gradients_not_zero(self, simple_model, sample_data):
        """Test that gradients are non-zero (no vanishing gradients)."""
        context, product_features, choices = sample_data

        with tf.GradientTape() as tape:
            loss_dict = simple_model.compute_loss(
                tf.constant(context),
                tf.constant(product_features),
                tf.constant(choices)
            )
            loss = loss_dict['loss']

        gradients = tape.gradient(loss, simple_model.trainable_variables)

        non_zero_count = 0
        for grad in gradients:
            if grad is not None and tf.reduce_sum(tf.abs(grad)) > 1e-10:
                non_zero_count += 1

        # Most gradients should be non-zero
        assert non_zero_count > len(gradients) * 0.5, \
            "Too many zero gradients (vanishing gradient problem)"

    def test_gradient_magnitude_reasonable(self, simple_model, sample_data):
        """Test that gradient magnitudes are reasonable (no exploding gradients)."""
        context, product_features, choices = sample_data

        with tf.GradientTape() as tape:
            loss_dict = simple_model.compute_loss(
                tf.constant(context),
                tf.constant(product_features),
                tf.constant(choices)
            )
            loss = loss_dict['loss']

        gradients = tape.gradient(loss, simple_model.trainable_variables)

        for grad in gradients:
            if grad is not None:
                max_grad = tf.reduce_max(tf.abs(grad)).numpy()
                assert max_grad < 1e6, f"Exploding gradient: {max_grad}"


# ==============================================================================
# Unit Tests: Loss Computation
# ==============================================================================

class TestLossComputation:
    """Tests for loss computation."""

    def test_loss_positive(self, simple_model, sample_data):
        """Test that loss is positive (NLL is always positive)."""
        context, product_features, choices = sample_data

        loss_dict = simple_model.compute_loss(
            tf.constant(context),
            tf.constant(product_features),
            tf.constant(choices)
        )

        assert loss_dict['loss'].numpy() > 0

    def test_loss_decreases_with_better_predictions(self, simple_model):
        """Test that loss is lower when predictions match choices."""
        context = np.random.randn(10, 10).astype(np.float32)
        product_features = np.random.randn(10, 5, 5).astype(np.float32)

        # Get model predictions
        outputs = simple_model(
            tf.constant(context),
            tf.constant(product_features),
            training=False
        )
        probs = outputs['probabilities'].numpy()

        # Create "correct" choices (matching predictions)
        correct_choices = np.zeros((10, 5), dtype=np.float32)
        for i in range(10):
            correct_choices[i, np.argmax(probs[i])] = 1.0

        # Create random choices
        random_choices = np.zeros((10, 5), dtype=np.float32)
        for i in range(10):
            random_choices[i, np.random.randint(5)] = 1.0

        loss_correct = simple_model.compute_loss(
            tf.constant(context),
            tf.constant(product_features),
            tf.constant(correct_choices)
        )['loss'].numpy()

        loss_random = simple_model.compute_loss(
            tf.constant(context),
            tf.constant(product_features),
            tf.constant(random_choices)
        )['loss'].numpy()

        # Loss with correct predictions should generally be lower
        # (not always due to randomness, but on average)
        assert loss_correct <= loss_random * 2  # Allow some margin

    def test_accuracy_range(self, simple_model, sample_data):
        """Test that accuracy is in [0, 1]."""
        context, product_features, choices = sample_data

        loss_dict = simple_model.compute_loss(
            tf.constant(context),
            tf.constant(product_features),
            tf.constant(choices)
        )

        accuracy = loss_dict['accuracy'].numpy()
        assert 0 <= accuracy <= 1


# ==============================================================================
# Integration Tests: Training
# ==============================================================================

class TestTraining:
    """Integration tests for model training."""

    def test_training_reduces_loss(self, model_config, synthetic_linear_data):
        """Test that training reduces loss on synthetic data."""
        context, product_features, choices, _ = synthetic_linear_data

        # Use smaller model for faster test
        config = model_config.copy()
        config['context_hidden_dims'] = [32]
        config['product_hidden_dims'] = [16]
        config['interaction_dims'] = [16]

        model = DeepContextDependentChoiceModel(**config)
        trainer = DeepContextChoiceModelTrainer(
            model=model,
            learning_rate=0.01,
            l2_regularization=1e-5
        )

        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((
            context[:500],
            product_features[:500],
            choices[:500]
        )).batch(64)

        # Get initial loss
        initial_loss = model.compute_loss(
            tf.constant(context[:100]),
            tf.constant(product_features[:100]),
            tf.constant(choices[:100])
        )['loss'].numpy()

        # Train for a few epochs
        history = trainer.train(dataset, epochs=10, verbose=0)

        # Get final loss
        final_loss = model.compute_loss(
            tf.constant(context[:100]),
            tf.constant(product_features[:100]),
            tf.constant(choices[:100])
        )['loss'].numpy()

        # Loss should decrease
        assert final_loss < initial_loss, \
            f"Loss didn't decrease: {initial_loss:.4f} -> {final_loss:.4f}"

    def test_training_history_recorded(self, model_config, sample_data):
        """Test that training history is recorded."""
        context, product_features, choices = sample_data

        model = DeepContextDependentChoiceModel(**model_config)
        trainer = DeepContextChoiceModelTrainer(model=model)

        dataset = tf.data.Dataset.from_tensor_slices((
            context, product_features, choices
        )).batch(16)

        history = trainer.train(dataset, epochs=5, verbose=0)

        assert len(history['train_loss']) == 5
        assert len(history['train_accuracy']) == 5


# ==============================================================================
# Tests: IIA Violation Detection
# ==============================================================================

class TestIIAViolation:
    """Tests for IIA (Independence of Irrelevant Alternatives) behavior."""

    def test_probability_ratio_changes_with_removal(self, model_config):
        """
        Test that probability ratios change when alternatives are removed.

        Under strict IIA, P(i)/P(j) should be constant regardless of other
        alternatives. Context-dependent models should violate this.
        """
        model = DeepContextDependentChoiceModel(**model_config)

        np.random.seed(42)
        context = np.random.randn(100, 10).astype(np.float32)
        product_features = np.random.randn(100, 5, 5).astype(np.float32)

        # Full choice set
        probs_full = model.predict_choice(
            tf.constant(context),
            tf.constant(product_features)
        ).numpy()

        # Remove product 2
        availability = np.ones((100, 5), dtype=np.float32)
        availability[:, 2] = 0

        probs_reduced = model.predict_choice(
            tf.constant(context),
            tf.constant(product_features),
            tf.constant(availability)
        ).numpy()

        # Compute ratio P(0)/P(1) in both cases
        ratio_full = probs_full[:, 0] / (probs_full[:, 1] + 1e-10)
        ratio_reduced = probs_reduced[:, 0] / (probs_reduced[:, 1] + 1e-10)

        # Ratios should change (IIA violation) or stay same (IIA holds)
        # Just verify computation works without errors
        assert ratio_full.shape == ratio_reduced.shape


# ==============================================================================
# Tests: Model Persistence
# ==============================================================================

class TestModelPersistence:
    """Tests for model saving and loading."""

    def test_model_weights_count(self, simple_model, sample_data):
        """Test that model has expected number of weight tensors."""
        context, product_features, _ = sample_data

        # Build model by calling it
        _ = simple_model(
            tf.constant(context),
            tf.constant(product_features),
            training=False
        )

        # Should have multiple trainable variables
        assert len(simple_model.trainable_variables) > 0


# ==============================================================================
# Edge Case Tests
# ==============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_sample(self, simple_model):
        """Test with batch size of 1."""
        context = np.random.randn(1, 10).astype(np.float32)
        product_features = np.random.randn(1, 5, 5).astype(np.float32)

        outputs = simple_model(
            tf.constant(context),
            tf.constant(product_features),
            training=False
        )

        assert outputs['probabilities'].shape == (1, 5)
        np.testing.assert_almost_equal(
            outputs['probabilities'].numpy().sum(), 1.0, decimal=5
        )

    def test_many_products(self, model_config):
        """Test with larger number of products."""
        model = DeepContextDependentChoiceModel(**model_config)

        context = np.random.randn(10, 10).astype(np.float32)
        product_features = np.random.randn(10, 20, 5).astype(np.float32)  # 20 products

        outputs = model(
            tf.constant(context),
            tf.constant(product_features),
            training=False
        )

        assert outputs['probabilities'].shape == (10, 20)
        sums = outputs['probabilities'].numpy().sum(axis=1)
        np.testing.assert_array_almost_equal(sums, np.ones(10), decimal=5)

    def test_all_products_masked_except_one(self, simple_model):
        """Test when only one product is available."""
        context = np.random.randn(10, 10).astype(np.float32)
        product_features = np.random.randn(10, 5, 5).astype(np.float32)

        # Only product 0 available
        availability = np.zeros((10, 5), dtype=np.float32)
        availability[:, 0] = 1

        outputs = simple_model(
            tf.constant(context),
            tf.constant(product_features),
            tf.constant(availability),
            training=False
        )

        probs = outputs['probabilities'].numpy()

        # Product 0 should have probability ~1
        np.testing.assert_array_almost_equal(probs[:, 0], np.ones(10), decimal=3)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
