import torch
import torch.nn as nn
from typing import Optional, Tuple

class SiglipVisionConfig:

    def __init__(
            self,
            hidden_size=768,
            intermediate_size=3072,
            num_hidden_layers=12,
            num_attention_head=12,
            num_channels=3,
            image_size=224,
            patch_size=16,
            layer_norm_eps=12-6,
            attention_dropout=0,
            num_image_token: int = None
    ):
        self.hidden_size = hidden_size
        self.intermediate_size= intermediate_size
        self.num_hidden_layers= num_hidden_layers
        self.num_attention_head = num_attention_head
        self.num_channels= num_channels
        self.image_size = image_size
        self.patch_size=patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout=attention_dropout
        self.num_image_token = num_image_token


class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        hidden_states = self.fc2(hidden_states)

        return hidden_states

class SiglipAttention(nn.Module):
    
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = self.config.hidden_size
        self.num_heads = self.config.num_attention_head
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = self.config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Lienar(self.embed_dim, self.embed_dim)
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len = hidden_states.size() # (batch_size, num_patches, embed_dim)

        query_states = self.q_proj(hidden_states) #(batch_size, num_patches, embed_dim)
        key_states = self.k_proj(hidden_states)#(batch_size, num_patches, embed_dim)
        value_states = self.v_proj(hidden_states) #(batch_size, num_patches, embed_dim)

        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attention_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale # batch_size, num_heads, num_patches, num_patches


        attention_weights = nn.functional.softmax(attention_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attention_weights = nn.functional.dropout(attention_weights, p=self.dropout, training=self.training)
        attention_output = torch.matmul(attention_weights, value_states) #batch_size, num_heads, num_patches, head_dim

        attention_output = attention_output.transpose(1, 2).contiguous() 
        attention_output = attention_output.reshape(batch_size, seq_len, self.embed_dim) 
        attention_output = self.out_proj(attention_output)

        return attention_output, attention_weights




class SiglipEncoder(nn.Module):
    
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.eps = config.layer_norm_eps
        self.attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=self.eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=self.eps)
    

    def forward(
            self,
            hidden_state: torch.Tensor
    ) -> torch.Tensor:
        residual = hidden_state #[Batch_Size, Num_Patches, Embed_Dim]
        hidden_state = self.layer_norm1(hidden_state) #[Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_state, _ = self.attn(hidden_state=hidden_state) # Batch_Size, Num_Patches, Embed_Dim]
        hidden_state = residual + hidden_state

        residual = hidden_state
        hidden_state = self.layer_norm2(hidden_state) #[Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim
        hidden_state = self.mlp(hidden_state)

        hidden_state = residual + hidden_state
        return hidden_state


class SiglipVisionEmbeddings(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        self.config = config
        self.embed_dim = self.config.hidden_size
        self.image_size = self.config.image_size
        self.patch_size = self.config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=self.config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid"
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1,-1)),
            persistant=False
        )
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape #(batch_size, num_channels, height, width)

        # Num_Patches_H = height // patch_size and Num_Patches_W = width // patch_size
        # Num_Patches = Num_Patches_H * Num_Patches_W
        patch_embeds = self.patch_embedding(pixel_values) # (batch_size, Embed_Dim, Num_Patches_H,  Num_Patches_W)
        embeddings = patch_embeds.flatten(2) # (batch_size, Embed_Dim,Num_Patches) 
        embeddings = embeddings.transpose(1, 2) # (batch_size, Num_Patches, Embed_Dim) 
        embeddings =  embeddings + self.position_embedding(self.position_ids) # [Batch_Size, Num_Patches, Embed_Dim] adding learned position embedding
        return embeddings


class SiglipVisionTransformer(nn.Module):
    
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = self.config.hidden_size
        eps = self.config.layer_norm_eps

        self.embedding = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=eps)
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:

        hidden_states = self.embedding(pixel_values) #(batch_size, num_channels, height, width) -> (batch_size, num_patches, embed_dim)
        hidden_states = self.encoder(input_embeds=hidden_states)
        hidden_states = self.post_layernorm(hidden_states)
        return hidden_states

class SiglipVisionModel(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)
    
    def forward(self, pixel_values) -> Tuple:
        # (batch_size, num_chanels, height width) -> (batch_size, num_patches, embed_dim)
        return self.vision_model(pixel_values=pixel_values)

