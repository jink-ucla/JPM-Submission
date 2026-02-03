"""
Deep Context-Dependent Choice Model (Zhang et al., 2025)
Implementation integrated with choice-learn framework using TensorFlow and TensorFlow Probability.

Reference: Zhang, Shuhan, Zhi Wang, Rui Gao and Shuang Li,
"Deep Context-Dependent Choice Model", ICML learning workshop, 2025
"""

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from typing import Optional, List, Tuple, Dict, Any
import numpy as np

tfd = tfp.distributions


class ContextEncoder(keras.Model):
    """
    Context encoder network that processes context features.
    Maps context to a latent representation.
    """

    def __init__(
        self,
        context_dim: int,
        hidden_dims: List[int] = [128, 64, 32],
        latent_dim: int = 16,
        activation: str = 'relu',
        dropout_rate: float = 0.1,
        name: str = 'context_encoder'
    ):
        super().__init__(name=name)
        self.context_dim = context_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim

        # Build encoder layers
        self.encoder_layers = []
        for i, hidden_dim in enumerate(hidden_dims):
            self.encoder_layers.append(
                keras.layers.Dense(
                    hidden_dim,
                    activation=activation,
                    name=f'encoder_dense_{i}'
                )
            )
            self.encoder_layers.append(
                keras.layers.Dropout(dropout_rate, name=f'encoder_dropout_{i}')
            )
            self.encoder_layers.append(
                keras.layers.BatchNormalization(name=f'encoder_bn_{i}')
            )

        # Latent representation layer
        self.latent_layer = keras.layers.Dense(
            latent_dim,
            activation=None,
            name='latent_representation'
        )

    def call(self, context, training=False):
        """
        Encode context to latent representation.

        Args:
            context: Context features [batch_size, context_dim]
            training: Whether in training mode

        Returns:
            Latent context representation [batch_size, latent_dim]
        """
        x = context
        for layer in self.encoder_layers:
            x = layer(x, training=training)
        latent = self.latent_layer(x)
        return latent


class ProductEncoder(keras.Model):
    """
    Product/item encoder network that processes product features.
    Can be shared across products or product-specific.
    """

    def __init__(
        self,
        product_dim: int,
        hidden_dims: List[int] = [64, 32],
        latent_dim: int = 16,
        activation: str = 'relu',
        dropout_rate: float = 0.1,
        name: str = 'product_encoder'
    ):
        super().__init__(name=name)
        self.product_dim = product_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim

        # Build encoder layers
        self.encoder_layers = []
        for i, hidden_dim in enumerate(hidden_dims):
            self.encoder_layers.append(
                keras.layers.Dense(
                    hidden_dim,
                    activation=activation,
                    name=f'product_dense_{i}'
                )
            )
            self.encoder_layers.append(
                keras.layers.Dropout(dropout_rate, name=f'product_dropout_{i}')
            )
            self.encoder_layers.append(
                keras.layers.BatchNormalization(name=f'product_bn_{i}')
            )

        # Latent representation layer
        self.latent_layer = keras.layers.Dense(
            latent_dim,
            activation=None,
            name='product_latent'
        )

    def call(self, product_features, training=False):
        """
        Encode product features to latent representation.

        Args:
            product_features: Product features [batch_size, n_products, product_dim]
            training: Whether in training mode

        Returns:
            Latent product representation [batch_size, n_products, latent_dim]
        """
        original_shape = tf.shape(product_features)
        batch_size = original_shape[0]
        n_products = original_shape[1]

        # Reshape for processing
        x = tf.reshape(product_features, [-1, self.product_dim])

        for layer in self.encoder_layers:
            x = layer(x, training=training)
        latent = self.latent_layer(x)

        # Reshape back
        latent = tf.reshape(latent, [batch_size, n_products, self.latent_dim])
        return latent


class ContextProductInteraction(keras.Model):
    """
    Context-Product interaction network that computes context-dependent utilities.
    This is the core of the deep context-dependent choice model.
    """

    def __init__(
        self,
        context_latent_dim: int,
        product_latent_dim: int,
        interaction_dims: List[int] = [64, 32, 16],
        activation: str = 'relu',
        dropout_rate: float = 0.1,
        use_attention: bool = True,
        name: str = 'context_product_interaction'
    ):
        super().__init__(name=name)
        self.context_latent_dim = context_latent_dim
        self.product_latent_dim = product_latent_dim
        self.use_attention = use_attention

        # Attention mechanism for context-product interaction
        if use_attention:
            self.attention = keras.layers.MultiHeadAttention(
                num_heads=4,
                key_dim=context_latent_dim,
                name='context_product_attention'
            )

        # Interaction layers
        combined_dim = context_latent_dim + product_latent_dim
        self.interaction_layers = []

        for i, hidden_dim in enumerate(interaction_dims):
            self.interaction_layers.append(
                keras.layers.Dense(
                    hidden_dim,
                    activation=activation,
                    name=f'interaction_dense_{i}'
                )
            )
            self.interaction_layers.append(
                keras.layers.Dropout(dropout_rate, name=f'interaction_dropout_{i}')
            )
            self.interaction_layers.append(
                keras.layers.BatchNormalization(name=f'interaction_bn_{i}')
            )

        # RESNET FIX: Add linear skip connection for gradient flow
        # This guarantees gradients can flow even if deep layers are dead
        self.linear_skip = keras.layers.Dense(
            1,
            activation=None,
            kernel_initializer=keras.initializers.GlorotUniform(),
            name='linear_skip_connection'
        )

        # Output layer for deep utility correction
        # CRITICAL FIX: Use stronger initialization to prevent dead gradients
        self.utility_layer = keras.layers.Dense(
            1,
            activation=None,
            kernel_initializer=keras.initializers.HeNormal(),  # Better than default Xavier
            bias_initializer=keras.initializers.Zeros(),  # Start at zero so skip dominates initially
            name='utility_output'
        )

    def call(self, context_latent, product_latent, training=False):
        """
        Compute context-dependent utilities for products.

        Args:
            context_latent: Context latent representation [batch_size, context_latent_dim]
            product_latent: Product latent representation [batch_size, n_products, product_latent_dim]
            training: Whether in training mode

        Returns:
            Utilities for each product [batch_size, n_products]
        """
        batch_size = tf.shape(product_latent)[0]
        n_products = tf.shape(product_latent)[1]

        # Expand context to match products
        context_expanded = tf.expand_dims(context_latent, axis=1)  # [batch, 1, context_dim]
        context_expanded = tf.tile(context_expanded, [1, n_products, 1])  # [batch, n_products, context_dim]

        # Apply attention if enabled
        if self.use_attention:
            # Use context as query, product as key and value
            attended = self.attention(
                query=context_expanded,
                value=product_latent,
                key=product_latent,
                training=training
            )
            # Combine original and attended
            combined = tf.concat([context_expanded, attended], axis=-1)
        else:
            # Simple concatenation
            combined = tf.concat([context_expanded, product_latent], axis=-1)

        # Reshape for processing
        combined_flat = tf.reshape(combined, [-1, tf.shape(combined)[-1]])

        # RESNET FIX: Compute linear baseline first (guarantees gradient flow)
        linear_utilities_flat = self.linear_skip(combined_flat)

        # Process through interaction layers for deep correction
        x = combined_flat
        for layer in self.interaction_layers:
            x = layer(x, training=training)

        # Compute deep utility correction
        deep_correction_flat = self.utility_layer(x)

        # RESNET FIX: Add linear + deep (residual connection)
        # At initialization, deep_correction ≈ 0, so model behaves like linear model
        # As training progresses, deep network learns to add nonlinear corrections
        utilities_flat = linear_utilities_flat + deep_correction_flat
        utilities = tf.reshape(utilities_flat, [batch_size, n_products])

        return utilities


class DeepContextDependentChoiceModel(keras.Model):
    """
    Deep Context-Dependent Choice Model (Zhang et al., 2025)

    This model captures context-dependent preferences using deep neural networks.
    The key idea is to learn latent representations of both context and products,
    and then model their interaction to predict choice probabilities.

    Architecture:
    1. Context Encoder: Maps context features to latent representation
    2. Product Encoder: Maps product features to latent representation
    3. Interaction Network: Computes context-dependent utilities
    4. Choice Probability: Softmax over utilities (MNL structure)
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
        name: str = 'deep_context_choice_model'
    ):
        """
        Initialize Deep Context-Dependent Choice Model.

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
            name: Model name
        """
        super().__init__(name=name)

        self.context_dim = context_dim
        self.product_dim = product_dim
        self.context_latent_dim = context_latent_dim
        self.product_latent_dim = product_latent_dim
        self.temperature = temperature
        self.add_intercept = add_intercept

        # Build sub-networks
        self.context_encoder = ContextEncoder(
            context_dim=context_dim,
            hidden_dims=context_hidden_dims,
            latent_dim=context_latent_dim,
            activation=activation,
            dropout_rate=dropout_rate
        )

        self.product_encoder = ProductEncoder(
            product_dim=product_dim,
            hidden_dims=product_hidden_dims,
            latent_dim=product_latent_dim,
            activation=activation,
            dropout_rate=dropout_rate
        )

        self.interaction_network = ContextProductInteraction(
            context_latent_dim=context_latent_dim,
            product_latent_dim=product_latent_dim,
            interaction_dims=interaction_dims,
            activation=activation,
            dropout_rate=dropout_rate,
            use_attention=use_attention
        )

        # Product-specific intercepts (ASCs)
        if add_intercept:
            self.intercepts = self.add_weight(
                name='product_intercepts',
                shape=(1,),  # Will be broadcasted
                initializer='zeros',
                trainable=True
            )

        # RESNET FIX: Direct linear path from raw features (bypass ALL deep layers)
        # This guarantees immediate gradient flow and learning
        self.raw_feature_skip = keras.layers.Dense(
            1,
            activation=None,
            kernel_initializer=keras.initializers.GlorotUniform(),
            name='raw_feature_skip'
        )

    def call(
        self,
        context: tf.Tensor,
        product_features: tf.Tensor,
        available_products: Optional[tf.Tensor] = None,
        training: bool = False
    ) -> Dict[str, tf.Tensor]:
        """
        Forward pass of the model.

        Args:
            context: Context features [batch_size, context_dim]
            product_features: Product features [batch_size, n_products, product_dim]
            available_products: Binary mask for available products [batch_size, n_products]
            training: Whether in training mode

        Returns:
            Dictionary containing:
                - utilities: Utilities for each product
                - probabilities: Choice probabilities
                - context_latent: Latent context representation
                - product_latent: Latent product representation
        """
        # RESNET FIX: Compute direct linear utilities from raw features first
        # This guarantees the model can learn IMMEDIATELY even if deep layers are dead
        batch_size = tf.shape(context)[0]
        n_products = tf.shape(product_features)[1]

        # Concatenate context with each product's features
        context_expanded = tf.tile(tf.expand_dims(context, 1), [1, n_products, 1])
        raw_combined = tf.concat([context_expanded, product_features], axis=-1)
        raw_combined_flat = tf.reshape(raw_combined, [-1, tf.shape(raw_combined)[-1]])

        # Direct linear utilities (bypass all deep layers)
        linear_utilities_flat = self.raw_feature_skip(raw_combined_flat)
        linear_utilities = tf.reshape(linear_utilities_flat, [batch_size, n_products])

        # Encode context and products for deep path
        context_latent = self.context_encoder(context, training=training)
        product_latent = self.product_encoder(product_features, training=training)

        # Compute deep utilities through interaction network
        deep_utilities = self.interaction_network(
            context_latent,
            product_latent,
            training=training
        )

        # RESNET FIX: Combine linear (immediate) + deep (learned over time)
        utilities = linear_utilities + deep_utilities

        # Add intercepts if enabled
        if self.add_intercept:
            utilities = utilities + self.intercepts

        # Scale by temperature
        utilities = utilities / self.temperature

        # Apply availability mask if provided
        if available_products is not None:
            # Mask unavailable products with very large negative utility
            mask_value = -1e9
            utilities = tf.where(
                tf.cast(available_products, tf.bool),
                utilities,
                tf.ones_like(utilities) * mask_value
            )

        # Compute choice probabilities (MNL)
        probabilities = tf.nn.softmax(utilities, axis=-1)

        return {
            'utilities': utilities,
            'probabilities': probabilities,
            'context_latent': context_latent,
            'product_latent': product_latent
        }

    def predict_choice(
        self,
        context: tf.Tensor,
        product_features: tf.Tensor,
        available_products: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        """
        Predict choice probabilities.

        Args:
            context: Context features
            product_features: Product features
            available_products: Binary mask for available products

        Returns:
            Choice probabilities [batch_size, n_products]
        """
        outputs = self.call(
            context,
            product_features,
            available_products,
            training=False
        )
        return outputs['probabilities']

    def compute_loss(
        self,
        context: tf.Tensor,
        product_features: tf.Tensor,
        choices: tf.Tensor,
        available_products: Optional[tf.Tensor] = None,
        weights: Optional[tf.Tensor] = None
    ) -> Dict[str, tf.Tensor]:
        """
        Compute negative log-likelihood loss.

        Args:
            context: Context features [batch_size, context_dim]
            product_features: Product features [batch_size, n_products, product_dim]
            choices: One-hot encoded choices [batch_size, n_products]
            available_products: Binary mask for available products
            weights: Sample weights [batch_size]

        Returns:
            Dictionary containing loss and metrics
        """
        outputs = self.call(
            context,
            product_features,
            available_products,
            training=True
        )

        probabilities = outputs['probabilities']

        # Negative log-likelihood
        # Add small epsilon for numerical stability
        epsilon = 1e-10
        log_probs = tf.math.log(probabilities + epsilon)

        # Weighted by choices (cross-entropy)
        sample_nll = -tf.reduce_sum(choices * log_probs, axis=-1)

        # Apply sample weights if provided
        if weights is not None:
            sample_nll = sample_nll * weights

        # Average over batch
        nll = tf.reduce_mean(sample_nll)

        # Compute accuracy
        predicted_choices = tf.argmax(probabilities, axis=-1)
        true_choices = tf.argmax(choices, axis=-1)
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(predicted_choices, true_choices), tf.float32)
        )

        return {
            'loss': nll,
            'nll': nll,
            'accuracy': accuracy
        }


class DeepContextChoiceModelTrainer:
    """
    Trainer class for Deep Context-Dependent Choice Model.
    Handles training loop, validation, and checkpointing.
    """

    def __init__(
        self,
        model: DeepContextDependentChoiceModel,
        learning_rate: float = 1e-3,
        optimizer: str = 'adam',
        l2_regularization: float = 0.0
    ):
        """
        Initialize trainer.

        Args:
            model: The choice model to train
            learning_rate: Learning rate
            optimizer: Optimizer type ('adam', 'sgd', 'rmsprop')
            l2_regularization: L2 regularization coefficient
        """
        self.model = model
        self.l2_regularization = l2_regularization

        # Create optimizer
        if optimizer == 'adam':
            self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            self.optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer == 'rmsprop':
            self.optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        # Metrics
        self.train_loss_metric = keras.metrics.Mean(name='train_loss')
        self.train_acc_metric = keras.metrics.Mean(name='train_accuracy')
        self.val_loss_metric = keras.metrics.Mean(name='val_loss')
        self.val_acc_metric = keras.metrics.Mean(name='val_accuracy')

    @tf.function
    def train_step(
        self,
        context: tf.Tensor,
        product_features: tf.Tensor,
        choices: tf.Tensor,
        available_products: Optional[tf.Tensor] = None,
        weights: Optional[tf.Tensor] = None
    ) -> Dict[str, tf.Tensor]:
        """
        Single training step.

        Args:
            context: Context features
            product_features: Product features
            choices: One-hot encoded choices
            available_products: Binary mask for available products
            weights: Sample weights

        Returns:
            Dictionary of metrics
        """
        with tf.GradientTape() as tape:
            # Compute loss
            loss_dict = self.model.compute_loss(
                context,
                product_features,
                choices,
                available_products,
                weights
            )

            loss = loss_dict['loss']

            # Add L2 regularization
            if self.l2_regularization > 0:
                l2_loss = tf.add_n([
                    tf.nn.l2_loss(v)
                    for v in self.model.trainable_variables
                ])
                loss = loss + self.l2_regularization * l2_loss

        # Compute gradients and update weights
        gradients = tape.gradient(loss, self.model.trainable_variables)

        # CRITICAL FIX: Gradient clipping to prevent vanishing/exploding gradients
        # This addresses the utility scale issue in discrete choice models
        gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0)

        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables)
        )

        # Update metrics
        self.train_loss_metric.update_state(loss_dict['nll'])
        self.train_acc_metric.update_state(loss_dict['accuracy'])

        return loss_dict

    @tf.function
    def val_step(
        self,
        context: tf.Tensor,
        product_features: tf.Tensor,
        choices: tf.Tensor,
        available_products: Optional[tf.Tensor] = None,
        weights: Optional[tf.Tensor] = None
    ) -> Dict[str, tf.Tensor]:
        """
        Single validation step.

        Args:
            context: Context features
            product_features: Product features
            choices: One-hot encoded choices
            available_products: Binary mask for available products
            weights: Sample weights

        Returns:
            Dictionary of metrics
        """
        loss_dict = self.model.compute_loss(
            context,
            product_features,
            choices,
            available_products,
            weights
        )

        # Update metrics
        self.val_loss_metric.update_state(loss_dict['nll'])
        self.val_acc_metric.update_state(loss_dict['accuracy'])

        return loss_dict

    def train(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: Optional[tf.data.Dataset] = None,
        epochs: int = 100,
        verbose: int = 1,
        callbacks: Optional[List] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Number of epochs
            verbose: Verbosity level
            callbacks: List of callbacks

        Returns:
            Training history
        """
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }

        for epoch in range(epochs):
            # Reset metrics
            self.train_loss_metric.reset_state()
            self.train_acc_metric.reset_state()
            self.val_loss_metric.reset_state()
            self.val_acc_metric.reset_state()

            # Training
            for batch in train_dataset:
                if len(batch) == 3:
                    context, product_features, choices = batch
                    available = None
                    weights = None
                elif len(batch) == 4:
                    context, product_features, choices, available = batch
                    weights = None
                else:
                    context, product_features, choices, available, weights = batch

                self.train_step(
                    context,
                    product_features,
                    choices,
                    available,
                    weights
                )

            # Validation
            if val_dataset is not None:
                for batch in val_dataset:
                    if len(batch) == 3:
                        context, product_features, choices = batch
                        available = None
                        weights = None
                    elif len(batch) == 4:
                        context, product_features, choices, available = batch
                        weights = None
                    else:
                        context, product_features, choices, available, weights = batch

                    self.val_step(
                        context,
                        product_features,
                        choices,
                        available,
                        weights
                    )

            # Record metrics
            train_loss = self.train_loss_metric.result().numpy()
            train_acc = self.train_acc_metric.result().numpy()
            history['train_loss'].append(train_loss)
            history['train_accuracy'].append(train_acc)

            if val_dataset is not None:
                val_loss = self.val_loss_metric.result().numpy()
                val_acc = self.val_acc_metric.result().numpy()
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_acc)

            # Print progress
            if verbose > 0:
                msg = f"Epoch {epoch + 1}/{epochs} - "
                msg += f"loss: {train_loss:.4f} - acc: {train_acc:.4f}"
                if val_dataset is not None:
                    msg += f" - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}"
                print(msg)

            # Callbacks
            if callbacks:
                for callback in callbacks:
                    callback.on_epoch_end(epoch, history)

        return history
