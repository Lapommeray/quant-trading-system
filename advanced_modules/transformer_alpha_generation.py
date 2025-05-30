import numpy as np
import pandas as pd
import logging
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim

class TimeSeriesTransformer(nn.Module):
    """
    Transformer-based model for financial time series forecasting
    """
    def __init__(self, input_size=10, output_size=1, d_model=64, nhead=4, num_layers=4, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.d_model = d_model
        
        self.embedding = nn.Linear(input_size, d_model)
        
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.decoder = nn.Linear(d_model, output_size)
        
    def forward(self, src):
        src = self.embedding(src) * np.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output[-1])  # Take the last output for prediction
        return output
        
class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer model
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
        
class TransformerAlphaGeneration:
    """
    Alpha generation using transformer-based models
    """
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.models = {}
        self.feature_columns = []
        self.target_column = 'returns'
        self.seq_length = 20
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def prepare_data(self, data, asset, feature_columns=None, target_column='returns', seq_length=20):
        """
        Prepare data for transformer model
        """
        if not isinstance(data, pd.DataFrame):
            self.logger.error("Data must be a pandas DataFrame")
            return None, None
            
        try:
            if feature_columns:
                self.feature_columns = feature_columns
            elif not self.feature_columns:
                self.feature_columns = ['open', 'high', 'low', 'close', 'volume']
                
            self.target_column = target_column
            self.seq_length = seq_length
            
            for col in self.feature_columns + [self.target_column]:
                if col not in data.columns:
                    self.logger.error(f"Column {col} not found in data")
                    return None, None
                    
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[self.feature_columns].iloc[i:i+seq_length].values)
                y.append(data[self.target_column].iloc[i+seq_length])
                
            X = np.array(X)
            y = np.array(y)
            
            X_tensor = torch.tensor(X, dtype=torch.float32).permute(1, 0, 2)  # [seq_len, batch_size, input_size]
            y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # [batch_size, 1]
            
            return X_tensor, y_tensor
        except Exception as e:
            self.logger.error(f"Error preparing data: {str(e)}")
            return None, None
            
    def train_model(self, data, asset, feature_columns=None, target_column='returns', seq_length=20, 
                   d_model=64, nhead=4, num_layers=4, dropout=0.1, lr=0.001, epochs=100, batch_size=32):
        """
        Train transformer model for alpha generation
        """
        X, y = self.prepare_data(data, asset, feature_columns, target_column, seq_length)
        if X is None or y is None:
            return False
            
        try:
            model = TimeSeriesTransformer(
                input_size=len(self.feature_columns),
                output_size=1,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dropout=dropout
            ).to(self.device)
            
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            
            X = X.to(self.device)
            y = y.to(self.device)
            
            model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                output = model(X)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                
                if epoch % 10 == 0:
                    self.logger.info(f"Epoch {epoch}, Loss: {loss.item():.6f}")
                    
            self.models[asset] = {
                'model': model,
                'feature_columns': self.feature_columns,
                'target_column': self.target_column,
                'seq_length': self.seq_length,
                'trained_at': datetime.now()
            }
            
            return True
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            return False
            
    def predict(self, data, asset):
        """
        Generate predictions using trained model
        """
        if asset not in self.models:
            self.logger.error(f"No trained model found for {asset}")
            return None
            
        model_info = self.models[asset]
        model = model_info['model']
        feature_columns = model_info['feature_columns']
        seq_length = model_info['seq_length']
        
        try:
            if len(data) < seq_length:
                self.logger.error(f"Data length ({len(data)}) is less than sequence length ({seq_length})")
                return None
                
            last_sequence = data[feature_columns].iloc[-seq_length:].values
            
            X = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(1)  # [seq_len, 1, input_size]
            X = X.to(self.device)
            
            model.eval()
            with torch.no_grad():
                prediction = model(X)
                
            return prediction.item()
        except Exception as e:
            self.logger.error(f"Error generating prediction: {str(e)}")
            return None
            
    def generate_signals(self, data, asset, threshold=0.0):
        """
        Generate trading signals based on predictions
        """
        if asset not in self.models:
            self.logger.error(f"No trained model found for {asset}")
            return None
            
        model_info = self.models[asset]
        feature_columns = model_info['feature_columns']
        seq_length = model_info['seq_length']
        
        try:
            signals = []
            
            for i in range(seq_length, len(data)):
                sequence = data[feature_columns].iloc[i-seq_length:i].values
                X = torch.tensor(sequence, dtype=torch.float32).unsqueeze(1)  # [seq_len, 1, input_size]
                X = X.to(self.device)
                
                model_info['model'].eval()
                with torch.no_grad():
                    prediction = model_info['model'](X).item()
                    
                if prediction > threshold:
                    signal = 1  # Buy
                elif prediction < -threshold:
                    signal = -1  # Sell
                else:
                    signal = 0  # Hold
                    
                signals.append({
                    'date': data.index[i],
                    'prediction': prediction,
                    'signal': signal
                })
                
            return pd.DataFrame(signals)
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
            return None
            
    def save_model(self, asset, file_path):
        """
        Save trained model to file
        """
        if asset not in self.models:
            self.logger.error(f"No trained model found for {asset}")
            return False
            
        try:
            model_info = self.models[asset]
            
            torch.save({
                'model_state_dict': model_info['model'].state_dict(),
                'feature_columns': model_info['feature_columns'],
                'target_column': model_info['target_column'],
                'seq_length': model_info['seq_length'],
                'trained_at': model_info['trained_at']
            }, file_path)
            
            self.logger.info(f"Model for {asset} saved to {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            return False
            
    def load_model(self, asset, file_path):
        """
        Load trained model from file
        """
        try:
            checkpoint = torch.load(file_path)
            
            model = TimeSeriesTransformer(
                input_size=len(checkpoint['feature_columns']),
                output_size=1
            ).to(self.device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            
            self.models[asset] = {
                'model': model,
                'feature_columns': checkpoint['feature_columns'],
                'target_column': checkpoint['target_column'],
                'seq_length': checkpoint['seq_length'],
                'trained_at': checkpoint['trained_at']
            }
            
            self.logger.info(f"Model for {asset} loaded from {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return False
