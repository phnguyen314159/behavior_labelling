import torch
import torch.nn as nn

#TODO:sarosh pls implement
class WindowGRU(nn.Module):
    #implement the gru
    #input_dim=768 (sbert encoding dim), hidden_dim=256, output_dim=6 (bart label 1x6)    
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=6, num_layers=1):
        super(WindowGRU, self).__init__()
        
        #decay param
        self.alpha = nn.Parameter(torch.tensor([0.1])) 
        
        self.gru_cell = nn.GRUCell(input_size=input_dim, hidden_size=hidden_dim)
        #req for pass input : [8, 100, 768]
        # output: [8,100,6] 
        
         # The fully connected linear layer to map the GRU hidden states to our 6 labels
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim
        
    def forward(self, x, rel_pos):
        batch_size, seq_len, _ = x.size()
        device = x.device
        
        h = torch.zeros(batch_size, self.hidden_dim).to(device)
        outputs = []

        for t in range(seq_len):
            if t == 0:
                delta_t = rel_pos[:, t].unsqueeze(1) 
            else:
                #vect sub
                delta_t = (rel_pos[:, t] - rel_pos[:, t-1]).unsqueeze(1)
            
            #h_decayed = h * exp(-|alpha| * delta_t)
            h = h * torch.exp(-torch.abs(self.alpha) * delta_t)
            
            #let gru handle its gate, we just give sugestions of how much it remember at each step
            h = self.gru_cell(x[:, t, :], h)
            
            #[8,1,256]
            outputs.append(h.unsqueeze(1))

        gru_out = torch.cat(outputs, dim=1) # [8, 100, 256]
        return self.fc(gru_out) # [8, 100, 6]