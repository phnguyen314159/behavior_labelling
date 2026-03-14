import torch
import torch.nn as nn

#TODO:sarosh pls implement
class WindowGRU(nn.Module):
    #implement the gru
    #input_dim=768 (sbert encoding dim), hidden_dim=256, output_dim=6 (bart label 1x6)    
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=6, num_layers=1):
        super(WindowGRU, self).__init__()
        
        # The GRU layer
        # batch_first=True ensures we take in [batch_size, seq_len, input_dim]
        self.gru = nn.GRU(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers,
            batch_first=True 
        )    
        #implement how the forward pass looks like
        #i will implement helper funct to create sequence chunk for each pass
        #req for pass input : [8, 100, 768]
        # output: [8,100, 6] 
        
         # The fully connected linear layer to map the GRU hidden states to our 6 labels
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x expected shape: [batch_size, seq_len, input_dim] -> [8, 100, 768]
        
        # gru_out contains the hidden states for ALL time steps (the whole sequence of 100)
        # hidden contains just the very last hidden state (we don't need it for this mapping)
        gru_out, hidden = self.gru(x) 
        # gru_out shape is now: [8, 100, 256]
        
        # Pass the entire sequence of hidden states through the linear layer
        out = self.fc(gru_out)
        # out shape is now: [8, 100, 6]
        
        return out