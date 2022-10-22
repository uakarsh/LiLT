## Embedding Layer

import torch.nn as nn
import torch
from einops import rearrange

class Embedding(nn.Module):
  
  def __init__(self, 
               vocab_size : int = 50265,  ## RobertA's tokenizer.vocab_size -> 50265
               hidden_dim_t : int = 768,  ## hidden_dim_text -> 768
               hidden_dim_l : int = 768 // 6,  ## hidden_dim_layout -> 768 // 6 for each of the 6 coordinates
               max_x_coord : int = 1001,  ## X coordinate ranges from 0 to 1000
               max_y_coord : int = 1001,
               max_seq_len_t : int = 512,
               max_seq_len_l : int  = 512):  ## Y coordinate ranges from 0 to 1000
      
      super(Embedding, self).__init__()
      self.lang_embedding = nn.Embedding(
                                  num_embeddings = vocab_size,
                                  embedding_dim = hidden_dim_t
                                  )
      
      self.top_left_x_emb = nn.Embedding(num_embeddings = max_x_coord,embedding_dim = hidden_dim_l)
      self.top_left_y_emb = nn.Embedding(num_embeddings = max_y_coord,embedding_dim = hidden_dim_l)
      self.bottom_right_x_emb = nn.Embedding(num_embeddings = max_x_coord,embedding_dim = hidden_dim_l)
      self.bottom_right_y_emb = nn.Embedding(num_embeddings = max_y_coord,embedding_dim = hidden_dim_l)
      self.width_emb = nn.Embedding(num_embeddings = max_x_coord,embedding_dim = hidden_dim_l)
      self.height_emb = nn.Embedding(num_embeddings = max_y_coord,embedding_dim = hidden_dim_l)

      self.box_position_embeddings = nn.Embedding(num_embeddings = max_seq_len_l + 1, embedding_dim = 6 * hidden_dim_l)
      self.textual_position_embeddings = nn.Embedding(num_embeddings = max_seq_len_t + 1, embedding_dim = hidden_dim_t)

      # ## Layer Normalization, would be added as pre-normalization and post-normalization
      # self.ln_t = nn.LayerNorm(normalized_shape = hidden_dim_t)
      # self.ln_l = nn.LayerNorm(normalized_shape = 6*hidden_dim_l)


  def forward(self, tokenized_words, tokenized_bbox):

    ## Generating position Ids
    text_len, box_len = tokenized_words.shape[1], tokenized_bbox.shape[1]
    word_pos_ids = torch.arange(text_len).unsqueeze(0).to(tokenized_words.device)
    box_pos_ids = torch.arange(box_len).unsqueeze(0).to(tokenized_bbox.device)

    ## Using Embedding Table for extracting the correspoding features
    text_feature = self.lang_embedding(tokenized_words)
    top_left_x_feat = self.top_left_x_emb(tokenized_bbox[:, :, 0])
    top_left_y_feat = self.top_left_y_emb(tokenized_bbox[:, :, 1])
    bottom_right_x_feat = self.bottom_right_x_emb(tokenized_bbox[:, :, 2])
    bottom_right_y_feat = self.bottom_right_y_emb(tokenized_bbox[:, :, 3])
    width_feat = self.width_emb(tokenized_bbox[:, :, 4])
    height_feat = self.height_emb(tokenized_bbox[:, :, 5])

    ## Layout feature
    layout_feature = torch.cat(
        [top_left_x_feat,
         top_left_y_feat,
         bottom_right_x_feat,
         bottom_right_y_feat,
         width_feat,
         height_feat
         ],
        axis = -1
    )

    ## Generating positional embedding
    pos_emb_t = self.textual_position_embeddings(word_pos_ids)
    pos_emb_l = self.box_position_embeddings(box_pos_ids)

    ## Adding a positional encoding
    layout_feature = layout_feature + pos_emb_l
    text_feature = text_feature + pos_emb_t

    # ## Adding the layer normalization, would be added in the encoder part
    # layout_feature = self.ln_l(layout_feature)
    # text_feature = self.ln_t(text_feature)

    return {'layout_feature': layout_feature, 'text_feature': text_feature}


## Attention Layer

## Reference: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
class MultiModalAttentionLayer(nn.Module):

  def __init__(self, embed_dim : int = 768, 
               n_heads : int = 12,
               dim_head : int = 64,
               fine_tune : bool = False,
               dropout : float = 0.0
               ):
    super(MultiModalAttentionLayer, self).__init__()

    inner_dim = n_heads * dim_head
    self.n_heads = n_heads
    self.fine_tune = fine_tune

    self.proj_text_k = nn.Linear(in_features = embed_dim, out_features = inner_dim)  ## 768 -> 512
    self.proj_text_q = nn.Linear(in_features = embed_dim, out_features = inner_dim)
    self.proj_text_v = nn.Linear(in_features = embed_dim, out_features = inner_dim)

    self.proj_layout_k = nn.Linear(in_features = embed_dim, out_features = inner_dim)
    self.proj_layout_q = nn.Linear(in_features = embed_dim, out_features = inner_dim)
    self.proj_layout_v = nn.Linear(in_features = embed_dim, out_features = inner_dim)

    self.attend = nn.Softmax(dim = -1)
    self.scale = dim_head ** -0.5

    self.dropout = nn.Dropout(dropout)
    self.to_out_l = nn.Sequential(
            nn.Linear(inner_dim, embed_dim),
            nn.Dropout(dropout)
        )
    self.to_out_t = nn.Sequential(
            nn.Linear(inner_dim, embed_dim),
            nn.Dropout(dropout)
        )

  def forward(self, text_feature, layout_feature):

    query_vec_t = rearrange(self.proj_text_q(text_feature), 'b t (head k) -> head b t k', head=self.n_heads)  ## batch, 512, 768 -> 8, batch, 512, 64
    key_vec_t = rearrange(self.proj_text_k(text_feature), 'b t (head k) -> head b t k', head=self.n_heads)
    value_vec_t = rearrange(self.proj_text_v(text_feature), 'b t (head k) -> head b t k', head=self.n_heads)

    query_vec_l = rearrange(self.proj_layout_q(layout_feature), 'b t (head k) -> head b t k', head=self.n_heads)
    key_vec_l = rearrange(self.proj_layout_k(layout_feature), 'b t (head k) -> head b t k', head=self.n_heads)
    value_vec_l = rearrange(self.proj_layout_v(layout_feature), 'b t (head k) -> head b t k', head=self.n_heads)

    attn_t = torch.einsum('hblk,hbtk->hblt', query_vec_t, key_vec_t) * self.scale
    attn_l = torch.einsum('hblk,hbtk->hblt', query_vec_l, key_vec_l) * self.scale

    attn_tilde_t = attn_t + attn_l

    if self.fine_tune:
      attn_tilde_l = attn_l + attn_t
    else:
      attn_tilde_l = attn_l + attn_t.detach()

    text_attn_probs =  self.dropout(self.attend(attn_tilde_t))
    layout_attn_probs =  self.dropout(self.attend(attn_tilde_l))
    
    text_context = rearrange(torch.einsum('hblt,hbtv->hblv', text_attn_probs, value_vec_t), 'h b l k -> b l (h k)')
    layout_context = rearrange(torch.einsum('hblt,hbtv->hblv', layout_attn_probs, value_vec_l), 'h b l k -> b l (h k)')

    text_context = self.to_out_t(text_context)
    layout_context = self.to_out_l(layout_context)

    return {'layout_feature': layout_context, 'text_feature': text_context, 
            'layout_attention': attn_l,'textual_attention': attn_t}


## Constructing the Encoder Layer

class PreNorm(nn.Module):
    def __init__(self, dim, fn, eps = 1e-12):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps = eps)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class PreNormAttn(nn.Module):
    def __init__(self, dim, fn, eps = 1e-12):
        super().__init__()

        self.norm_t = nn.LayerNorm(dim, eps = eps)
        self.norm_l = nn.LayerNorm(dim, eps = eps)
        self.fn = fn

    def forward(self, text_feat, layout_feat, **kwargs):
        return self.fn(self.norm_t(text_feat),
                       self.norm_l(layout_feat),**kwargs)


## FFN Network
class FeedForward(nn.Module):
    def __init__(self, dim : int = 768, hidden_dim : int = 4 * 768, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


## Encoder
class LiLTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([])
        for _ in range(config['num_hidden_layers']):
            encoder_block = nn.ModuleList([
                PreNormAttn(config['hidden_size'],
                            MultiModalAttentionLayer(embed_dim = config['hidden_size'],
                                                     n_heads = config['num_attention_heads'],
                                                     dim_head = config['dim_head'],
                                                     fine_tune = config['fine_tune'],
                                                     dropout = config['hidden_dropout_prob'],
                                                     ),
                            eps = config['eps']
                            ),
                PreNorm(config['hidden_size'],
                        FeedForward(config['hidden_size'],
                                    config['hidden_size'] * config['intermediate_ff_size_factor'],
                                    dropout=config['hidden_dropout_prob'],
                        ),
                        eps = config['eps']),
                PreNorm(config['hidden_size'],
                        FeedForward(config['hidden_size'],
                                    config['hidden_size'] * config['intermediate_ff_size_factor'],
                                    dropout=config['hidden_dropout_prob']
                        ),
                        eps = config['eps'])
            ])
            self.layers.append(encoder_block)

    def forward(
            self,
            text_feat,
            layout_feat,
    ):

        text_attn = []
        layout_attn = []
        text_hidden_states = []
        layout_hidden_states = []

        for attn, ff_t, ff_l in self.layers:

          context_vec = attn(text_feat, layout_feat)
          text_feat = text_feat + context_vec['text_feature']
          layout_feat = layout_feat + context_vec['layout_feature']

          text_feat = ff_t(text_feat) + text_feat
          layout_feat = ff_l(layout_feat) + layout_feat

          text_attn.append(context_vec['textual_attention'])
          layout_attn.append(context_vec['layout_attention'])
          text_hidden_states.append(text_feat)
          layout_hidden_states.append(layout_feat)

        return {'text_hidden_states' : text_hidden_states, 'layout_hidden_states': layout_hidden_states,
                'text_attn' : text_attn, 'layout_attn' : layout_attn}



## Constructing the whole model from embeddings to the hidden states and attention
class LiLT(nn.Module):

  def __init__(self, config):
    super(LiLT, self).__init__()
    self.lilt = LiLTEncoder(config)
    self.emb = Embedding(vocab_size = config['vocab_size'],
                hidden_dim_t = config['hidden_size_t'],
                hidden_dim_l = config['hidden_size_l'],
                max_x_coord = config['max_2d_position_embeddings'],
                max_y_coord = config['max_2d_position_embeddings'],
                max_seq_len_t = config['max_seq_len_t'],
                max_seq_len_l = config['max_seq_len_l'])
    

  def forward(self, tokenized_words, tokenized_bbox):
    hidden_enc = self.emb(tokenized_words, tokenized_bbox)
    encodings = self.lilt(hidden_enc['text_feature'], hidden_enc['layout_feature'])
    return encodings
