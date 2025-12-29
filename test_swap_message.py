import torch
# messages shape (2* 2, 1) = (num_envs*num_agnents, T)
s_message = torch.tensor([0,1,0,1,0,1,0,2,0,2,0,8,0,9,0,10,0,11])
s_msgs_reshaped = s_message.view(-1, 2)  # Shape: (Num_Envs, 2)
swapped_msgs = torch.flip(s_msgs_reshaped, dims=[1]).flatten() # Shape: (Total_Agents,)

print(swapped_msgs)