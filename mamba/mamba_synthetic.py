import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba.modules.mamba2_simple import Mamba2Simple

class SyntheticEHRMamba(nn.Module):
    def __init__(self, code_num=2869, d_model=256, n_layer=4, d_state=64, d_conv=4):
        super().__init__()
        self.code_num = code_num
        self.d_model = d_model
        
        print(f"Initializing Mamba model with:")
        print(f"  Code num: {code_num}")
        print(f"  d_model: {d_model}")
        print(f"  n_layer: {n_layer}")
        print(f"  d_state: {d_state}")
        
        # 1. Multi-hot input projection - QUAN TRỌNG: thay thế embedding
        self.input_proj = nn.Linear(code_num, d_model)
        
        # 2. Mamba2Simple layers
        self.mamba_layers = nn.ModuleList([
            Mamba2Simple(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=2,
                headdim=64,  # Giảm để nhẹ hơn
                chunk_size=64,              # Thêm dòng này (giảm từ 256 → 64)
                use_mem_eff_path=False,
                layer_idx=i,
            ) for i in range(n_layer)
        ])
        
        # 3. Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # 4. Multi-label prediction head với sigmoid
        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, code_num),
            nn.Sigmoid()  # For multi-label probabilities
        )
        
    def forward(self, x):
        """
        x: (batch_size, seq_len, code_num) - multi-hot vectors
        returns: (batch_size, seq_len, code_num) - prediction probabilities
        """
        # Project multi-hot to hidden dimension
        hidden_states = self.input_proj(x)  # (batch, seq_len, d_model)
        
        # Pass through Mamba layers
        for layer in self.mamba_layers:
            hidden_states = layer(hidden_states)  # (batch, seq_len, d_model)
        
        # Normalize and predict
        hidden_states = self.norm(hidden_states)
        probabilities = self.pred_head(hidden_states)  # (batch, seq_len, code_num)
        
        return probabilities

class SyntheticMambaTrainer:
    def __init__(self, model, learning_rate=1e-3):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()  # For probabilities
        
    def train_step(self, batch_data):
        """
        batch_data: (x, lens) 
        - x: (batch, max_visits, code_num) multi-hot
        - lens: (batch,) actual sequence lengths
        """
        x, lens = batch_data
        
        # Next-visit prediction: predict visit t+1 from visits 0..t
        inputs = x[:, :-1]   # All visits except last (0 to n-2)
        targets = x[:, 1:]   # All visits except first (1 to n-1)
        
        # Forward pass
        self.optimizer.zero_grad()
        predictions = self.model(inputs)  # Already probabilities due to sigmoid
        
        # Calculate loss
        loss = self.criterion(predictions, targets)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate_batch(self, batch_data):
        """Evaluate on a single batch"""
        x, lens = batch_data
        inputs = x[:, :-1]
        targets = x[:, 1:]
        
        with torch.no_grad():
            predictions = self.model(inputs)
            loss = self.criterion(predictions, targets)
            
        return loss.item(), predictions, targets
    
    def predict_next_visit(self, patient_sequence):
        """
        Predict next visit for a single patient sequence
        patient_sequence: (num_visits, code_num)
        returns: (code_num,) - predicted probabilities for next visit
        """
        self.model.eval()
        with torch.no_grad():
            # Add batch dimension
            sequence = patient_sequence.unsqueeze(0)  # (1, num_visits, code_num)
            
            # Get predictions for all positions
            predictions = self.model(sequence)  # (1, num_visits, code_num)
            
            # Get prediction for the next visit (last position)
            next_visit_probs = predictions[0, -1]  # (code_num,)
            
        return next_visit_probs
