import torch
import torch.nn as nn


shoe_sizes = torch.tensor([[6.0], [8.0], [10.0]])
actual_heights = torch.tensor([[24.0], [32.0], [40.0]])

model = nn.Linear(in_features=1, out_features=1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

for round in range(100):
    predictions = model(shoe_sizes)                     
    error = ((predictions - actual_heights) ** 2).mean()  
    
    optimizer.zero_grad()  
    error.backward()       
    optimizer.step()       

print(error.item())  