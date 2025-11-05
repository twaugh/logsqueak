tags:: #machine-learning, #deep-learning
type:: framework
language:: [[Python]]
created:: [[2024-02-12]]

- TensorFlow is a deep learning framework
  - Developed by Google Brain team
  - Open-sourced in 2015
  - Used for [[Machine Learning]] at scale
- Core concepts
  id:: 65a2c1f1-7777-4567-89ab-cdef01234587
  - Tensors
    - Multi-dimensional arrays
    - Flow through computation graph
  - Operations (ops)
    - Mathematical operations on tensors
  - Graphs
    - Computational workflow
  - Sessions (TF 1.x)
    - Deprecated in TF 2.x
- TensorFlow 2.x
  - Eager execution by default
  - Keras integrated as high-level API
  - Simpler and more Pythonic
  - tf.function for graph optimization
- Keras API
  - Sequential model (simple)
  - Functional API (flexible)
  - Model subclassing (advanced)
  - Built-in layers and activations
- Training workflow
  - Define model architecture
  - Compile with optimizer and loss
  - Fit on training data
  - Evaluate on test data
  - Make predictions
- Advanced features
  - Custom layers and models
  - Custom training loops
  - Callbacks for monitoring
  - TensorBoard for visualization
  - Distributed training
- Deployment
  - TensorFlow Serving
  - TensorFlow Lite (mobile)
  - TensorFlow.js (browser)
  - SavedModel format
- vs [[PyTorch]]
  - TensorFlow: production-ready
  - PyTorch: research-friendly
  - Both excellent choices
- Ecosystem
  - TensorFlow Hub (pre-trained models)
  - TensorFlow Extended (TFX for production)
  - TensorFlow Probability
- Use cases
  - Image classification
  - Natural language processing
  - Time series forecasting
  - Recommender systems
