# ML Models

This directory contains machine learning models trained using Google Colab for the QMP Overrider system. These models are used by the QuantConnect algorithms to make trading decisions.

## Models

- `liquidity_predictor.pkl`: Predicts the impact of dark pool trades on the market
- `hft_behavior.h5`: Predicts HFT reactions to order book imbalances

## Training

These models are trained using Google Colab notebooks in the `/colab_training` directory. The training process is as follows:

1. Run the Colab notebook to train the model
2. Save the model to a file
3. Push the model file to GitHub
4. QuantConnect pulls the latest model from GitHub

## Model Performance Logging

The system includes a Model Performance Logger that tracks model drift over time. This allows for continuous improvement of the models based on real-world performance.

## Integration with QuantConnect

The models are loaded by the QuantConnect algorithms using the `self.Download()` method, which pulls the latest model from GitHub. This allows for seamless updates to the models without having to redeploy the algorithms.

## Example Usage

```python
# In QuantConnect algorithm
def Initialize(self):
    self.model = joblib.load(self.Download("github.com/yourrepo/ml_models/liquidity_predictor.pkl"))
```
