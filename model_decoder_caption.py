# model_decoder_caption.py
import torch
from torch import nn
import math
from torch.nn.init import xavier_uniform_
from torch import Tensor
from typing import Optional
from torch.nn import functional as F
from transformers import MambaConfig, MambaModel, GPT2Config, GPT2Model

# --- Reuse existing PositionalEncoding, Mesh_TransformerDecoderLayer, StackTransformer ---
# (Assuming these are defined in a shared file or copied here)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

        self.embedding_1D = nn.Embedding(52, int(d_model))
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Mesh_TransformerDecoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Mesh_TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(int(d_model), nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

        self.fc_alpha1 = nn.Linear(d_model + d_model, d_model)
        self.fc_alpha2 = nn.Linear(d_model + d_model, d_model)
        self.fc_alpha3 = nn.Linear(d_model + d_model, d_model)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_alpha1.weight)
        nn.init.xavier_uniform_(self.fc_alpha2.weight)
        nn.init.xavier_uniform_(self.fc_alpha3.weight)
        nn.init.constant_(self.fc_alpha1.bias, 0)
        nn.init.constant_(self.fc_alpha2.bias, 0)
        nn.init.constant_(self.fc_alpha3.bias, 0)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        self_att_tgt = self.norm1(tgt + self._sa_block(tgt, tgt_mask, tgt_key_padding_mask))
        enc_att, att_weight = self._mha_block(self_att_tgt, memory, memory_mask, memory_key_padding_mask)
        x = self.norm2(self_att_tgt + enc_att)
        x = self.norm3(x + self._ff_block(x))
        return x + tgt

    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout1(x)

    def _mha_block(self, x: Tensor, mem: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x, att_weight = self.multihead_attn(x, mem, mem, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=True)
        return self.dropout2(x), att_weight

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

class StackTransformer(nn.Module):
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(StackTransformer, self).__init__()
        self.layers = torch.nn.modules.transformer._get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

# --- Caption Decoder ---
class CaptionDecoderTransformer(nn.Module):
    def __init__(self, decoder_type, embed_dim, vocab_size, max_lengths, word_vocab, n_head, n_layers, dropout):
        super(CaptionDecoderTransformer, self).__init__()

        print("caption_decoder_n_layers=", n_layers)

        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.max_lengths = max_lengths
        self.word_vocab = word_vocab
        self.dropout = dropout
        self.vocab_embedding = nn.Embedding(vocab_size, self.embed_dim)
        self.decoder_type = decoder_type
        print("decoder_type=", self.decoder_type)

        if self.decoder_type == 'mamba':
            config_1 = MambaConfig(num_hidden_layers=1, hidden_size=embed_dim)
            self.Mamba = nn.ModuleList([MambaModel(config_1) for _ in range(n_layers)])
        elif self.decoder_type == 'gpt':
            config_2 = GPT2Config(n_layer=1, n_embd=embed_dim)
            self.GPT = nn.ModuleList([GPT2Model(config_2) for _ in range(n_layers)])
        else:
            decoder_layer = Mesh_TransformerDecoderLayer(embed_dim, n_head, dim_feedforward=embed_dim * 4, dropout=dropout)
            self.transformer = StackTransformer(decoder_layer, n_layers)

        self.position_encoding = PositionalEncoding(embed_dim, max_len=max_lengths)
        self.wdc = nn.Linear(embed_dim, vocab_size)
        self.dropout_layer = nn.Dropout(p=self.dropout)

        self.init_weights()

    def init_weights(self):
        self.vocab_embedding.weight.data.uniform_(-0.1, 0.1)
        self.wdc.bias.data.fill_(0)
        self.wdc.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, encoded_captions, caption_lengths):
        """
        :param x: encoded features from encoder_trans, shape: (batch_size, channel, 49)
        :param encoded_captions: a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: a tensor of dimension (batch_size)
        """

        batch, channel, flat_hw = x.size()
        spatial_size = int(flat_hw ** 0.5) # Calculate spatial size (e.g., 7 for 49)

        # --- Text Generation Path ---
        # Reshape x from [B, C, 49] to [49, B, C] for transformer
        x_flat_seq = x.view(batch, channel, spatial_size, spatial_size).view(batch, channel, -1).permute(2, 0, 1) # [49, B, C]

        word_length = encoded_captions.size(1)
        mask = torch.triu(torch.ones(word_length, word_length) * float('-inf'), diagonal=1).cuda()
        tgt_pad_mask = (encoded_captions == self.word_vocab['<NULL>']) | (encoded_captions == self.word_vocab['<END>'])

        word_emb = self.vocab_embedding(encoded_captions).transpose(1, 0) # [L, B, embed_dim]
        word_emb = self.position_encoding(word_emb)

        if self.decoder_type in ['mamba', 'gpt']:
            img_emb = x_flat_seq.permute(1, 0, 2) # [B, 49, embed_dim]
            img_emb_len = img_emb.size(1)
            text = word_emb.permute(1, 0, 2) # [B, L, embed_dim]
            prefix = torch.cat((img_emb, text), dim=1)
            model_list = self.Mamba if self.decoder_type == 'mamba' else self.GPT
            for layer in model_list:
                prefix = layer(inputs_embeds=prefix).last_hidden_state
            pred = prefix[:, img_emb_len:, :].permute(1, 0, 2) # [B, L, embed_dim]
        else:
            pred = self.transformer(word_emb, x_flat_seq, tgt_mask=mask, tgt_key_padding_mask=tgt_pad_mask) # [L, B, embed_dim]
        
        pred = self.wdc(self.dropout_layer(pred)).permute(1, 0, 2) # [B, L, vocab_size]

        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        encoded_captions = encoded_captions[sort_ind]
        pred = pred[sort_ind]
        decode_lengths = (caption_lengths - 1).tolist()
        # --- End Text Generation Path ---

        return pred, encoded_captions, decode_lengths, sort_ind

    def sample(self, x, k=1):
        # Similar logic as before, using x_flat_seq
        batch, channel, flat_hw = x.size()
        spatial_size = int(flat_hw ** 0.5)
        x_flat_seq = x.view(batch, channel, spatial_size, spatial_size).view(batch, channel, -1).permute(2, 0, 1) # [49, B, C]

        tgt = torch.zeros(batch, self.max_lengths).to(torch.int64).cuda()
        mask = torch.triu(torch.ones(self.max_lengths, self.max_lengths) * float('-inf'), diagonal=1).cuda()
        tgt[:, 0] = torch.LongTensor([self.word_vocab['<START>']] * batch).cuda()
        seqs = torch.LongTensor([[self.word_vocab['<START>']]] * batch).cuda()

        for step in range(self.max_lengths):
            tgt_pad_mask = (tgt == self.word_vocab['<NULL>'])
            word_emb = self.vocab_embedding(tgt).transpose(1, 0)
            word_emb = self.position_encoding(word_emb)

            if self.decoder_type in ['mamba', 'gpt']:
                img_emb = x_flat_seq.permute(1, 0, 2)
                img_emb_len = img_emb.size(1)
                text = word_emb.permute(1, 0, 2)
                prefix = torch.cat((img_emb, text), dim=1)
                model_list = self.Mamba if self.decoder_type == 'mamba' else self.GPT
                for layer in model_list:
                    prefix = layer(inputs_embeds=prefix).last_hidden_state
                pred = prefix[:, img_emb_len:, :].permute(1, 0, 2)
            else:
                pred = self.transformer(word_emb, x_flat_seq, tgt_mask=mask, tgt_key_padding_mask=tgt_pad_mask)

            pred = self.wdc(self.dropout_layer(pred)).permute(1, 0, 2)
            scores = pred[:, step, :].squeeze(1)
            predicted_id = torch.argmax(scores, axis=-1)
            seqs = torch.cat([seqs, predicted_id.unsqueeze(1)], dim=-1)

            if predicted_id == self.word_vocab['<END>']:
                break
            if step < (self.max_lengths - 1):
                tgt[:, step + 1] = predicted_id
        seqs = seqs.squeeze(0)
        seqs = seqs.tolist()
        return seqs

    def sample_beam(self, x, k=3):
        # Similar logic as before, using x_flat_seq
        batch, channel, flat_hw = x.size()
        spatial_size = int(flat_hw ** 0.5)
        x_spatial = x.view(batch, channel, spatial_size, spatial_size)
        x_flat_seq = x_spatial.view(batch, channel, -1).unsqueeze(0).expand(k, -1, -1, -1).reshape(batch*k, channel, flat_hw).permute(2, 0, 1) # [49, B*k, C]

        tgt = torch.zeros(k*batch, self.max_lengths).to(torch.int64).cuda()
        mask = (torch.triu(torch.ones(self.max_lengths, self.max_lengths)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).cuda()
        tgt[:, 0] = torch.LongTensor([self.word_vocab['<START>']] * batch * k).cuda()
        seqs = torch.LongTensor([[self.word_vocab['<START>']]] * batch * k).cuda()
        top_k_scores = torch.zeros(k*batch, 1).cuda()
        complete_seqs = []
        complete_seqs_scores = []

        for step in range(self.max_lengths):
            word_emb = self.vocab_embedding(tgt).transpose(1, 0)
            word_emb = self.position_encoding(word_emb)

            if self.decoder_type in ['mamba', 'gpt']:
                img_emb = x_flat_seq.permute(1, 0, 2)
                img_emb_len = img_emb.size(1)
                text = word_emb.permute(1, 0, 2)
                prefix = torch.cat((img_emb, text), dim=1)
                model_list = self.Mamba if self.decoder_type == 'mamba' else self.GPT
                for layer in model_list:
                    prefix = layer(inputs_embeds=prefix).last_hidden_state
                pred = prefix[:, img_emb_len:, :].permute(1, 0, 2)
            else:
                pred = self.transformer(word_emb, x_flat_seq, tgt_mask=mask)

            pred = self.wdc(self.dropout_layer(pred)).permute(1, 0, 2)
            scores = pred[:, step, :].squeeze(1)
            scores = F.log_softmax(scores, dim=1)
            scores = top_k_scores.expand_as(scores) + scores

            if step == 0:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)

            prev_word_inds = torch.div(top_k_words, self.vocab_size, rounding_mode='floor')
            next_word_inds = top_k_words % self.vocab_size
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=-1)

            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != self.word_vocab['<END>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])

            k -= len(complete_inds)
            if k == 0:
                break

            seqs = seqs[incomplete_inds]
            x_flat_seq = x_flat_seq[:, prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            tgt = tgt[incomplete_inds]
            if step < self.max_lengths - 1:
                tgt[:, :step+2] = seqs

        if not complete_seqs:
            complete_seqs.extend(seqs[incomplete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[incomplete_inds])

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        return seq

    def fine_tune(self, fine_tune=True):
        for p in self.parameters():
            p.requires_grad = fine_tune