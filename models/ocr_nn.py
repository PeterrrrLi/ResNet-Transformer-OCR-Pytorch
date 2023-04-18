# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    ocr_nn.py                                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: peterli <j2457li@uwaterloo.ca>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/04/18 17:20:33 by peterli           #+#    #+#              #
#    Updated: 2023/04/18 17:20:35 by peterli          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from torch import nn
from torchvision.models import resnet18
import torch
from einops import rearrange


class SelfAttention(nn.Module):
    
    '''
    Multi-Head Self Attention

    Args:
        embed_dim: number of features of word vector
        num_head: number of heads
        is_masked: Whether to show mask, if true, then network would only be able to see first

    Shape:
        - Input: Batch, Series, Vector
        - Output: Batch Series, Vector

    Examples::
        # >>> m = SelfAttention(720, 12)
        # >>> x = torch.randn(4, 13, 720)
        # >>> output = m(x)
        # >>> print(output.shape)
        # torch.Size([4, 13, 720])
    '''

    def __init__(self, embed_dim, num_head, is_masked=True):
        super(SelfAttention, self).__init__()
        assert embed_dim % num_head == 0
        self.num_head = num_head
        self.is_masked = is_masked
        self.linear1 = nn.Linear(embed_dim, 3 * embed_dim)
        self.linear2 = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        
        '''
        x has shape Batch, Series, Vector
        '''
        
        # Shape become Batch, Series, 3 * Vector 
        x = self.linear1(x)
        n, s, v = x.shape
        
        # Shape become Batch, Series, H, Vector 
        x = x.reshape(n, s, self.num_head, -1)
        
        # Shape become Batch, H, Series, Vector
        x = torch.transpose(x, 1, 2)
        
        query, key, value = torch.chunk(x, 3, -1)
        dk = value.shape[-1] ** 0.5
        
        # Self Attention
        w = torch.matmul(query, key.transpose(-1, -2)) / dk 
        
        if self.is_masked:
            mask = torch.tril(torch.ones(w.shape[-1], w.shape[-1])).to(w.device)
            w = w * mask - 1e10 * (1 - mask)
        
        # Combine for result
        w = torch.softmax(w, dim=-1)
        attention = torch.matmul(w, value)
        attention = attention.permute(0, 2, 1, 3)
        n, s, h, v = attention.shape
        
        # Concatinate
        attention = attention.reshape(n, s, h * v)
        
        # Linear Layer
        return self.linear2(attention)


class Block(nn.Module):
    
    '''
    Block

    Args:
        embed_dim: number of features of word vector
        num_head: number of heads
        is_masked: Whether to show mask, if true, then network would only be able to see first

    Shape:
        - Input: Batch, Series, Vector
        - Output: Batch Series, Vector

    Examples::
        # >>> m = Block(720, 12)
        # >>> x = torch.randn(4, 13, 720)
        # >>> output = m(x)
        # >>> print(output.shape)
        # torch.Size([4, 13, 720])
    '''

    def __init__(self, embed_dim, num_head, is_masked): 
        super(Block, self).__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attention = SelfAttention(embed_dim, num_head, is_masked)
        self.ln_2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 6),
            nn.ReLU(),
            nn.Linear(embed_dim * 6, embed_dim),
        )

    def forward(self, x):
        
        # First, get attention
        attention = self.attention(self.ln_1(x))
        
        # Residual
        x = attention + x
        x = self.ln_2(x)
        
        # Feed Forward
        h = self.feed_forward(x)
        x = h + x 
        return x


class AbsPosEmb(nn.Module):
    def __init__(self, fmap_size, dim_head):  # (3,9)  # 512
        super().__init__()
        height, width = fmap_size  # 3,9
        scale = dim_head**-0.5
        self.height = nn.Parameter(torch.randn(height, dim_head) * scale)
        self.width = nn.Parameter(torch.randn(width, dim_head) * scale)

    def forward(self):
        emb = rearrange(self.height, "h d -> h () d") + rearrange(
            self.width, "w d -> () w d"
        )
        emb = rearrange(emb, " h w d -> (w h) d")
        # logits = torch.einsum('b i d, j d -> b i j', q, emb)
        return emb


class OcrNet(nn.Module):
    
    '''
    Input Dimension: 3, 48, 144
    
    Ouput Dimension: 27, Batch, Num of Classes
    '''

    def __init__(self, num_class):
        super(OcrNet, self).__init__()
        resnet = resnet18(True)
        backbone = list(resnet.children())
        self.backbone = nn.Sequential(
            nn.BatchNorm2d(3),
            *backbone[:3],
            *backbone[4:8],
        )
        self.decoder = nn.Sequential(
            Block(512, 8, False),
            Block(512, 8, False),
            Block(512, 8, False),
        )
        self.out_layer = nn.Linear(512, num_class)
        self.abs_pos_emb = AbsPosEmb((3, 9), 512)

    def forward(self, x):
        x = self.backbone(x)
        x = rearrange(x, "n c h w -> n (w h) c")
        x = x + self.abs_pos_emb()
        x = self.decoder(x)
        x = rearrange(x, "n s v -> s n v")
        return self.out_layer(x)


if __name__ == "__main__":
    m = OcrNet(70)
    print(m)
    x = torch.randn(32, 3, 48, 144)
    print(m(x).shape)
