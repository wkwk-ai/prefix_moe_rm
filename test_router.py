import torch
from router.modeling import Router
best_model = Router()
best_model.load_state_dict(torch.load('/home/haowang/prefix_moe_rm/results/router/best_model.pt'))
best_model.eval()