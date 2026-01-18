import torch
import torch.nn as nn
import torchvision.transforms as T

import numpy as np
import os
from typing import Callable

from torch.distributions.categorical import Categorical


from efficientnet_pytorch import EfficientNet


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PPOLSTMCommAgent(nn.Module):
    '''
    Agent with communication
    Observations: [image, location, message]
    '''
    def __init__(self, num_actions=3, n_words=4, embedding_size=64, num_channels=3, image_size=96, max_duration=12,
    pretrained_embedding=False, freeze_embedding=False, embedding_path=None):
        super().__init__()
        self.max_duration = max_duration # maximum spawn durations: initial training is 6 but we can extend to 12
        self.visual_dim = 128
        self.n_words = n_words
        self.embedding_size = embedding_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.loc_dim = 64 # location encoder output size

        # RNN hyperparameters
        self.input_dim = self.visual_dim + self.loc_dim + self.embedding_size # RNN input dim
        self.hidden_dim = 256 # RNN hidden dim
        

        
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.visual_encoder = EfficientNetEncoder() # output is (B, 1280)
        self.visual_projector = nn.Sequential(
            nn.Linear(1280, self.visual_dim),
            nn.ReLU()
        )

        self.message_encoder = nn.Sequential(
            nn.Embedding(n_words+1, embedding_size), # +1 means it includes silence token that is not produced by the model
            nn.Linear(embedding_size, embedding_size), 
            nn.ReLU(),
        )
        if pretrained_embedding:
            # 1. Load the state dict into a variable first
            checkpoint = torch.load(embedding_path)
            
            # 1. Extract and rename only the keys meant for the message_encoder
            prefix = "message_encoder."
            loaded_state_dict = {
                k[len(prefix):]: v for k, v in checkpoint.items() 
                if k.startswith(prefix)
            }
            
            # 2. Load into the model and capture the return values
            load_result = self.message_encoder.load_state_dict(loaded_state_dict, strict=False)
            
            # 3. Calculate statistics
            total_model_keys = len(self.message_encoder.state_dict())
            total_loaded_keys = len(loaded_state_dict)
            missing_count = len(load_result.missing_keys)
            unexpected_count = len(load_result.unexpected_keys)
            
            # The number of keys that successfully matched is the total in the model minus what was missing
            matched_count = total_model_keys - missing_count
            
            print(f"Total keys in model: {total_model_keys}")
            print(f"Total keys in loaded dict: {total_loaded_keys}")
            print(f"Matched keys: {matched_count}")
            print(f"Missing keys (in model but not in dict): {missing_count}")
            print(f"Unexpected keys (in dict but not in model): {unexpected_count}")

        if freeze_embedding:
            for param in self.message_encoder.parameters():
                param.requires_grad = False
                
        self.location_encoder = nn.Linear(4, self.loc_dim) # Input size 4 (x, y , dx, dy)
        

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim)
        
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        self.actor = layer_init(nn.Linear(self.hidden_dim, num_actions), std=0.01)
        self.critic = layer_init(nn.Linear(self.hidden_dim, 1), std=1)
        self.message_head = layer_init(nn.Linear(self.hidden_dim, n_words), std=0.01)

        self.mask_head = layer_init(nn.Linear(self.hidden_dim, 2), std=0.01)
        self.time_head = layer_init(nn.Linear(self.hidden_dim, self.max_duration), std=0.01)

        # self-prediction head: predict the next latent state
        self.predictor = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.input_dim)
            )
    def _get_features(self, input):
        """
        Helper to extract features from raw observations.
        Used for both the current step (input to LSTM) and the next step (target for predictor).
        """
        image, location, message = input

        x = image / 255.0
        x = self.normalize(x)
        image_feat = self.visual_projector(self.visual_encoder(x).flatten(start_dim=1))

        location_feat = self.location_encoder(location)

        message_feat = self.message_encoder(message) 
        message_feat = message_feat.view(-1, self.embedding_size)

        features = torch.cat((image_feat, location_feat, message_feat), dim=1)
        return features

    def get_states(self, input, lstm_state, done, tracks=None):
        batch_size = lstm_state[0].shape[1]
        hidden = self._get_features(input)
    
        # LSTM logic
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        if tracks is not None:
            tracks = tracks.reshape((-1, batch_size)) # Not used in logic but kept for interface
            
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, input, lstm_state, done, action=None, message=None, tracks=None):
        image, location, received_message = input
        hidden, lstm_state = self.get_states((image, location, received_message), lstm_state, done, tracks)

        action_logits = self.actor(hidden)
        action_probs = Categorical(logits=action_logits)
        
        if action is None:
            action = action_probs.sample()

        message_logits = self.message_head(hidden)
        message_probs = Categorical(logits=message_logits)
        
        if message is None:
            # Inference
            # Shift message to [1, n_words]
            raw_message_idx = message_probs.sample()
            message = raw_message_idx + 1
        else:
            # Training
            # Shift message back to [0, n_words-1]
            raw_message_idx = (message - 1).clamp(min=0)

        mask_logits = self.mask_head(hidden)
        time_logits = self.time_head(hidden)

        predicted_next_features = self.predictor(hidden)

        return (
            action, 
            action_probs.log_prob(action), 
            action_probs.entropy(), 
            message, 
            message_probs.log_prob(raw_message_idx), 
            message_probs.entropy(), 
            self.critic(hidden), 
            mask_logits,
            time_logits,
            predicted_next_features,
            lstm_state
        )
        


class EfficientNetEncoder(nn.Module):
    '''
    Use pretrained EfficientNetEncoder from NoMaD navigation model
    '''
    def __init__(self, model_name='efficientnet-b0', pretrained_path="checkpoints/visual_encoder/pretrained_nomad.pth", image_size=(96,96), train_visual_encoder=False):
        super().__init__()
        
        # 1. Initialize Base Structure
        self.base_model = EfficientNet.from_name(model_name)
        self.base_model = replace_bn_with_gn(self.base_model)
        
        # 2. Load Weights from Nomad Checkpoint (If provided)
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"Loading pretrained weights from {pretrained_path}...")
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # Use 'state_dict' if it's nested, otherwise assume the dict itself is weights
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                checkpoint = checkpoint['model_state_dict']

            # Filter and Rename Keys
            new_state_dict = {}
            for k, v in checkpoint.items():
                # --- FIX: UPDATED PREFIX MATCHING ---
                if k.startswith("vision_encoder.obs_encoder."):
                    # Remove the full prefix so it matches the base_model keys
                    new_key = k.replace("vision_encoder.obs_encoder.", "")
                    new_state_dict[new_key] = v
            
            missing, unexpected = self.base_model.load_state_dict(new_state_dict, strict=False)
            print(f"Weights loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        else:
            print("No pretrained path provided or file not found. Using random init.")

        if not train_visual_encoder:
            print("Freeze Visual Encoder Parameters")
            for param in self.base_model.parameters():
                param.requires_grad = False
        
    def forward(self, x):
        x = self.base_model.extract_features(x) # get encoding of this img 
        x = self.base_model._avg_pooling(x) # avg pooling 
        return x

def replace_bn_with_gn(root_module: nn.Module, features_per_group: int=16) -> nn.Module:
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=max(1, x.num_features//features_per_group), 
            num_channels=x.num_features)
    )
    return root_module

def replace_submodules(root_module: nn.Module, predicate: Callable[[nn.Module], bool], func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    if predicate(root_module):
        return func(root_module)
    bn_list = [k.split('.') for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    return root_module


if __name__ == "__main__":
    dummy_inp  = torch.randn(32,3,96,96)
    enc = EfficientNetEncoder()
    dummy_out = enc(dummy_inp)
    print(dummy_out.shape)


    agent = PPOLSTMCommAgent()