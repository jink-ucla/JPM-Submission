"""
Choice-Learn Wrapper for Deep Context-Dependent Choice Model
Integrates the Zhang et al. (2025) model with the choice-learn framework.
"""

import tensorflow as tf
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from .deep_context_choice_model import (
    DeepContextDependentChoiceModel,
    DeepContextChoiceModelTrainer
)

try:
    from choice_learn.models.base_model import ChoiceModel
    from choice_learn.data import ChoiceDataset
    CHOICE_LEARN_AVAILABLE = True
except ImportError:
    print("Warning: choice-learn not available. Using base implementation.")
    CHOICE_LEARN_AVAILABLE = False
    # Create dummy base class
    class ChoiceModel:
        pass


class DeepContextChoiceLearn(ChoiceModel if CHOICE_LEARN_AVAILABLE else object):
    """
    Choice-Learn compatible wrapper for Deep Context-Dependent Choice Model.

    This class provides an interface compatible with the choice-learn framework,
    allowing the deep context-dependent model to be used alongside other choice
    models in the library.
    """

    def __init__(
        self,
        context_dim: int,
        product_dim: int,
        context_hidden_dims: List[int] = [128, 64, 32],
        product_hidden_dims: List[int] = [64, 32],
        context_latent_dim: int = 16,
        product_latent_dim: int = 16,
        interaction_dims: List[int] = [64, 32, 16],
        activation: str = 'relu',
        dropout_rate: float = 0.1,
        use_attention: bool = True,
        temperature: float = 1.0,
        add_intercept: bool = True,
        learning_rate: float = 1e-3,
        optimizer: str = 'adam',
        l2_regularization: float = 0.0,
        **kwargs
    ):
        """
        Initialize the choice-learn compatible model.

        Args:
            context_dim: Dimension of context features
            product_dim: Dimension of product features
            context_hidden_dims: Hidden layer dimensions for context encoder
            product_hidden_dims: Hidden layer dimensions for product encoder
            context_latent_dim: Latent dimension for context
            product_latent_dim: Latent dimension for products
            interaction_dims: Hidden layer dimensions for interaction network
            activation: Activation function
            dropout_rate: Dropout rate
            use_attention: Whether to use attention mechanism
            temperature: Temperature parameter for softmax
            add_intercept: Whether to add product-specific intercepts
            learning_rate: Learning rate
            optimizer: Optimizer type
            l2_regularization: L2 regularization coefficient
        """
        if CHOICE_LEARN_AVAILABLE:
            super().__init__(**kwargs)

        self.context_dim = context_dim
        self.product_dim = product_dim
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.l2_regularization = l2_regularization

        # Create the model
        self.model = DeepContextDependentChoiceModel(
            context_dim=context_dim,
            product_dim=product_dim,
            context_hidden_dims=context_hidden_dims,
            product_hidden_dims=product_hidden_dims,
            context_latent_dim=context_latent_dim,
            product_latent_dim=product_latent_dim,
            interaction_dims=interaction_dims,
            activation=activation,
            dropout_rate=dropout_rate,
            use_attention=use_attention,
            temperature=temperature,
            add_intercept=add_intercept
        )

        # Create trainer
        self.trainer = DeepContextChoiceModelTrainer(
            model=self.model,
            learning_rate=learning_rate,
            optimizer=optimizer,
            l2_regularization=l2_regularization
        )

        self.history = None

    def fit(
        self,
        context: np.ndarray,
        product_features: np.ndarray,
        choices: np.ndarray,
        available_products: Optional[np.ndarray] = None,
        sample_weights: Optional[np.ndarray] = None,
        val_context: Optional[np.ndarray] = None,
        val_product_features: Optional[np.ndarray] = None,
        val_choices: Optional[np.ndarray] = None,
        val_available_products: Optional[np.ndarray] = None,
        batch_size: int = 128,
        epochs: int = 100,
        verbose: int = 1,
        **kwargs
    ):
        """
        Fit the model to training data.

        Args:
            context: Context features [n_samples, context_dim]
            product_features: Product features [n_samples, n_products, product_dim]
            choices: One-hot encoded choices [n_samples, n_products]
            available_products: Binary mask for available products [n_samples, n_products]
            sample_weights: Sample weights [n_samples]
            val_context: Validation context features
            val_product_features: Validation product features
            val_choices: Validation choices
            val_available_products: Validation availability mask
            batch_size: Batch size for training
            epochs: Number of training epochs
            verbose: Verbosity level
        """
        # Convert to tensors and create dataset
        if available_products is not None:
            if sample_weights is not None:
                train_dataset = tf.data.Dataset.from_tensor_slices((
                    context.astype(np.float32),
                    product_features.astype(np.float32),
                    choices.astype(np.float32),
                    available_products.astype(np.float32),
                    sample_weights.astype(np.float32)
                ))
            else:
                train_dataset = tf.data.Dataset.from_tensor_slices((
                    context.astype(np.float32),
                    product_features.astype(np.float32),
                    choices.astype(np.float32),
                    available_products.astype(np.float32)
                ))
        else:
            train_dataset = tf.data.Dataset.from_tensor_slices((
                context.astype(np.float32),
                product_features.astype(np.float32),
                choices.astype(np.float32)
            ))

        train_dataset = train_dataset.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # Create validation dataset if provided
        val_dataset = None
        if val_context is not None:
            if val_available_products is not None:
                val_dataset = tf.data.Dataset.from_tensor_slices((
                    val_context.astype(np.float32),
                    val_product_features.astype(np.float32),
                    val_choices.astype(np.float32),
                    val_available_products.astype(np.float32)
                ))
            else:
                val_dataset = tf.data.Dataset.from_tensor_slices((
                    val_context.astype(np.float32),
                    val_product_features.astype(np.float32),
                    val_choices.astype(np.float32)
                ))
            val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # Train the model
        self.history = self.trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=epochs,
            verbose=verbose
        )

        return self

    def predict_probabilities(
        self,
        context: np.ndarray,
        product_features: np.ndarray,
        available_products: Optional[np.ndarray] = None,
        batch_size: int = 1024
    ) -> np.ndarray:
        """
        Predict choice probabilities.

        Args:
            context: Context features [n_samples, context_dim]
            product_features: Product features [n_samples, n_products, product_dim]
            available_products: Binary mask for available products [n_samples, n_products]
            batch_size: Batch size for prediction

        Returns:
            Choice probabilities [n_samples, n_products]
        """
        n_samples = context.shape[0]
        all_probs = []

        # Predict in batches
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)

            batch_context = tf.constant(context[i:end_idx].astype(np.float32))
            batch_products = tf.constant(product_features[i:end_idx].astype(np.float32))
            batch_available = None
            if available_products is not None:
                batch_available = tf.constant(available_products[i:end_idx].astype(np.float32))

            batch_probs = self.model.predict_choice(
                batch_context,
                batch_products,
                batch_available
            )
            all_probs.append(batch_probs.numpy())

        return np.concatenate(all_probs, axis=0)

    def predict(
        self,
        context: np.ndarray,
        product_features: np.ndarray,
        available_products: Optional[np.ndarray] = None,
        batch_size: int = 1024
    ) -> np.ndarray:
        """
        Predict chosen alternatives.

        Args:
            context: Context features
            product_features: Product features
            available_products: Binary mask for available products
            batch_size: Batch size for prediction

        Returns:
            Predicted choice indices [n_samples]
        """
        probs = self.predict_probabilities(
            context,
            product_features,
            available_products,
            batch_size
        )
        return np.argmax(probs, axis=1)

    def evaluate(
        self,
        context: np.ndarray,
        product_features: np.ndarray,
        choices: np.ndarray,
        available_products: Optional[np.ndarray] = None,
        batch_size: int = 1024
    ) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            context: Context features
            product_features: Product features
            choices: One-hot encoded choices or choice indices
            available_products: Binary mask for available products
            batch_size: Batch size for evaluation

        Returns:
            Dictionary of evaluation metrics
        """
        # Get predictions
        probs = self.predict_probabilities(
            context,
            product_features,
            available_products,
            batch_size
        )

        # Convert choices if needed
        if choices.ndim == 1 or choices.shape[1] == 1:
            # Choice indices
            choice_indices = choices.flatten()
            choices_onehot = np.zeros_like(probs)
            choices_onehot[np.arange(len(choice_indices)), choice_indices] = 1
        else:
            # Already one-hot
            choices_onehot = choices
            choice_indices = np.argmax(choices, axis=1)

        # Compute accuracy
        predicted_indices = np.argmax(probs, axis=1)
        accuracy = np.mean(predicted_indices == choice_indices)

        # Compute negative log-likelihood
        epsilon = 1e-10
        nll = -np.mean(np.sum(choices_onehot * np.log(probs + epsilon), axis=1))

        # Compute Brier score
        brier_score = np.mean((probs - choices_onehot)**2)

        return {
            'accuracy': accuracy,
            'nll': nll,
            'brier_score': brier_score,
            'log_likelihood': -nll * len(context)
        }

    def save_weights(self, filepath: str):
        """Save model weights."""
        self.model.save_weights(filepath)

    def load_weights(self, filepath: str):
        """Load model weights."""
        self.model.load_weights(filepath)

    def get_latent_representations(
        self,
        context: np.ndarray,
        product_features: np.ndarray,
        batch_size: int = 1024
    ) -> Dict[str, np.ndarray]:
        """
        Get latent representations for context and products.

        Args:
            context: Context features
            product_features: Product features
            batch_size: Batch size for computation

        Returns:
            Dictionary containing latent representations
        """
        n_samples = context.shape[0]
        all_context_latent = []
        all_product_latent = []

        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)

            batch_context = tf.constant(context[i:end_idx].astype(np.float32))
            batch_products = tf.constant(product_features[i:end_idx].astype(np.float32))

            outputs = self.model(batch_context, batch_products, training=False)

            all_context_latent.append(outputs['context_latent'].numpy())
            all_product_latent.append(outputs['product_latent'].numpy())

        return {
            'context_latent': np.concatenate(all_context_latent, axis=0),
            'product_latent': np.concatenate(all_product_latent, axis=0)
        }


def convert_choicelearn_dataset(dataset: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a choice-learn ChoiceDataset to numpy arrays.

    Args:
        dataset: ChoiceDataset object

    Returns:
        Tuple of (context, product_features, choices)
    """
    if not CHOICE_LEARN_AVAILABLE:
        raise ImportError("choice-learn is not available")

    # Extract data from ChoiceDataset
    # This is a placeholder - actual implementation depends on ChoiceDataset structure
    context = dataset.get_context_features()
    product_features = dataset.get_product_features()
    choices = dataset.get_choices()

    return context, product_features, choices


# Example usage function
def example_usage():
    """Example of how to use the choice-learn wrapper."""
    print("Example: Deep Context Choice Model with Choice-Learn Interface")

    # Generate synthetic data
    n_samples = 1000
    context_dim = 10
    product_dim = 5
    n_products = 5

    context = np.random.randn(n_samples, context_dim)
    product_features = np.random.randn(n_samples, n_products, product_dim)

    # Generate synthetic choices
    true_utilities = np.random.randn(n_samples, n_products)
    probs = np.exp(true_utilities) / np.exp(true_utilities).sum(axis=1, keepdims=True)
    choices = np.zeros((n_samples, n_products))
    for i in range(n_samples):
        choices[i, np.random.choice(n_products, p=probs[i])] = 1

    # Create and fit model
    model = DeepContextChoiceLearn(
        context_dim=context_dim,
        product_dim=product_dim,
        context_hidden_dims=[64, 32],
        product_hidden_dims=[32, 16],
        context_latent_dim=12,
        product_latent_dim=12,
        interaction_dims=[48, 24],
        use_attention=True,
        learning_rate=1e-3
    )

    print("Fitting model...")
    model.fit(
        context=context,
        product_features=product_features,
        choices=choices,
        batch_size=64,
        epochs=20,
        verbose=1
    )

    # Evaluate
    metrics = model.evaluate(context, product_features, choices)
    print("\nEvaluation metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # Make predictions
    predictions = model.predict(context[:10], product_features[:10])
    print("\nPredictions (first 10):", predictions)

    # Get latent representations
    latent = model.get_latent_representations(context[:10], product_features[:10])
    print("\nContext latent shape:", latent['context_latent'].shape)
    print("Product latent shape:", latent['product_latent'].shape)


if __name__ == '__main__':
    example_usage()
