from typing import List
from typing import Tuple
import torch
import torch.nn as nn
from einops import rearrange
from torch import FloatTensor, LongTensor
import numpy as np
from Pos_Former.datamodule import vocab, vocab_size 
from Pos_Former.model.pos_enc import WordPosEnc
from Pos_Former.model.transformer.arm import AttentionRefinementModule
from Pos_Former.model.transformer.transformer_decoder import (
    TransformerDecoder,
    TransformerDecoderLayer,
)
from Pos_Former.utils.generation_utils import DecodeModel, PosDecodeModel


def _build_transformer_decoder(
    d_model: int,
    nhead: int,
    num_decoder_layers: int,
    dim_feedforward: int,
    dropout: float,
    dc: int,
    cross_coverage: bool,
    self_coverage: bool,
) -> nn.TransformerDecoder:
    decoder_layer = TransformerDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )
    if cross_coverage or self_coverage:
        arm = AttentionRefinementModule(nhead, dc, cross_coverage, self_coverage)
    else:
        arm = None

    decoder = TransformerDecoder(decoder_layer, num_decoder_layers, arm)
    return decoder

import torch
import torch.nn as nn

class FusionModule(nn.Module):
    def __init__(self,  d_model: int,):
        super(FusionModule, self).__init__()
        self.d_model = d_model
        self.w_att = nn.Linear(2 * d_model, d_model)

    def forward(self, e_feature: torch.FloatTensor, i_feature: torch.FloatTensor):
        """generate output fusing e_feature & i_feature

        Parameters
        ----------
        e_feature : FloatTensor
            [b, l, d]
        i_feature: FloatTensor
            [b, l, d]

        Returns
        -------
        FloatTensor
            [b, l, d]
        """
        f = torch.cat((e_feature, i_feature), dim=2)
        f_att = torch.sigmoid(self.w_att(f))
        output = f_att * i_feature + (1 - f_att) * e_feature
        return output
""" if __name__ == "__main__":
    # 创建一个示例输入张量 (batch_size, sequence_length, d_model)
    batch_size = 2
    sequence_length = 5
    d_model = 256
    e_feature = torch.randn(8, 4, 256)
    i_feature = torch.randn(8, 4, 256)

    # 初始化 FusionModule 模型
    model = FusionModule(d_model)

    # 前向传播
    output = model(e_feature, i_feature)

    # 打印输出特征
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}") """
class Decoder(DecodeModel):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        dc: int,
        cross_coverage: bool,
        self_coverage: bool,
    ):
        super().__init__()
        self.fusion = FusionModule(d_model=d_model)
        self.word_embed = nn.Sequential(
            nn.Embedding(vocab_size, d_model), nn.LayerNorm(d_model)
        )
        self.pos_enc = WordPosEnc(d_model=d_model)

        self.norm = nn.LayerNorm(d_model)

        self.model = _build_transformer_decoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
        )

        self.proj = nn.Linear(d_model, vocab_size)

    def _build_attention_mask(self, length):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.full(
            (length, length), fill_value=1, dtype=torch.bool, device=self.device
        )
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(
        self, src: FloatTensor, src_mask: LongTensor, tgt: LongTensor 
    ) -> FloatTensor:
        """generate output for tgt

        Parameters
        ----------
        src : FloatTensor
            [b, h, w, d]torch.Size([8, 4, 4, 256])
        src_mask: LongTensor
            [b, h, w]8, 4, 4,
        tgt : LongTensor
            [b, l] 1 8 18 0
        pos_labels:LongTensor
            [b,l,5]
        is_test:bool
        这段代码是一个Transformer模型中的前向传播（forward pass）部分，通常用于自然语言处理任务，如机器翻译或文本生成。下面我将逐步解释这段代码的含义、实现原理、用途和注意事项。

### 代码解释

1. **获取目标序列的长度**：
    ```python
    _, l = tgt.size()  # l=4 _=8
    ```
    这里，`tgt`是目标序列（target sequence），`tgt.size()`返回序列的形状。假设`tgt`的形状是`[batch_size, seq_length]`，那么`l`就是序列的长度（seq_length），而`_`是批大小（batch_size）。

2. **构建注意力掩码**：
    ```python
    tgt_mask = self._build_attention_mask(l)
    ```
    `self._build_attention_mask(l)`是一个方法，用于生成注意力掩码（attention mask），以防止模型在处理序列时“看到”未来的信息。

3. **生成填充掩码**：
    ```python
    tgt_pad_mask = tgt == vocab.PAD_IDX
    ```
    这行代码生成一个填充掩码（padding mask），用于标记目标序列中的填充位置（PAD tokens）。`vocab.PAD_IDX`是填充符号的索引。

4. **词嵌入和位置编码**：
    ```python
    tgt_vocab = tgt
    tgt = self.word_embed(tgt)  # [b, l, d]
    tgt = self.pos_enc(tgt)  # [b, l, d]
    tgt = self.norm(tgt)
    ```
    - `self.word_embed(tgt)`将目标序列中的每个词转换为词嵌入向量。
    - `self.pos_enc(tgt)`对词嵌入向量进行位置编码，以保留序列中词的位置信息。
    - `self.norm(tgt)`对嵌入向量进行归一化处理。

5. **重排源序列和掩码**：
    ```python
    h = src.shape[1]
    src = rearrange(src, "b h w d -> (h w) b d")
    src_mask = rearrange(src_mask, "b h w -> b (h w)")
    tgt = rearrange(tgt, "b l d -> l b d")
    ```
    - `h = src.shape[1]`获取源序列的高度（假设源序列的形状是`[batch_size, height, width, depth]`）。
    - `rearrange`函数用于重新排列张量的维度，使其适应Transformer模型的输入格式。

6. **模型前向传播**：
    ```python
    out, attn  = self.model(
        tgt=tgt,
        memory=src,
        height=h,
        tgt_mask=tgt_mask,
        tgt_key_padding_mask=tgt_pad_mask,
        memory_key_padding_mask=src_mask,
        tgt_vocab=tgt_vocab,
    )
    ```
    - `self.model`是Transformer模型，它接受目标序列`tgt`、源序列`src`、高度`h`、注意力掩码`tgt_mask`、目标填充掩码`tgt_key_padding_mask`、源填充掩码`memory_key_padding_mask`和目标词汇`tgt_vocab`作为输入。
    - `out`是模型的输出，`attn`是注意力权重。

7. **重排输出并投影**：
    ```python
    out_rearrange = rearrange(out, "l b d -> b l d")
    out = self.proj(out_rearrange)
    return out, attn
    ```
    - `rearrange(out, "l b d -> b l d")`将输出张量重新排列回`[batch_size, seq_length, depth]`的形状。
    - `self.proj(out_rearrange)`对输出进行线性变换（投影），以得到最终的输出。
    - 最后，返回模型的输出`out`和注意力权重`attn`。

### 注意事项

1. **掩码的作用**：
    - 注意力掩码（attention mask）用于防止模型在解码时“看到”未来的信息。
    - 填充掩码（padding mask）用于忽略填充位置，使模型只关注有效的序列元素。

2. **词嵌入和位置编码**：
    - 词嵌入将词转换为向量表示，位置编码保留词的位置信息，两者结合使模型能够理解序列中的词和它们的相对位置。

3. **维度变换**：
    - 在Transformer模型中，输入和输出张量的维度需要经过多次变换，以适应模型的输入和输出格式。

4. **模型架构**：
    - 这段代码假设使用的是Transformer模型，具体实现可能因应用场景和模型架构的不同而有所变化。

通过这段代码，我们可以看到Transformer模型在处理序列数据时的基本流程，包括词嵌入、位置编码、注意力机制和线性变换等步骤。
        Returns
        -------
        FloatTensor
            [b, l, vocab_size]
        """
        _, l = tgt.size()#l=4 _=8
        tgt_mask = self._build_attention_mask(l)
        tgt_pad_mask = tgt == vocab.PAD_IDX

        tgt_vocab=tgt
        tgt = self.word_embed(tgt)  # [b, l, d]
        tgt = self.pos_enc(tgt)  # [b, l, d]
        tgt = self.norm(tgt)
        
        h = src.shape[1]
        src = rearrange(src, "b h w d -> (h w) b d")
        src_mask = rearrange(src_mask, "b h w -> b (h w)")
        print(tgt.shape)
        tgt_test = tgt
        tgt = rearrange(tgt, "b l d -> l b d")
        
        out, attn  = self.model(
            tgt=tgt,
            memory=src,
            height=h,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_mask,
            tgt_vocab=tgt_vocab,
        )
    
        out_rearrange = rearrange(out, "l b d -> b l d")
        print("out_rearrange",out_rearrange.shape)
        print("out",attn.shape)
        #fusion_out = self.fusion(tgt_test, out_rearrange)
        print("fusion_out",fusion_out.shape)
        out = self.proj(fusion_out)
        return out, attn

    def transform(
        self, src: List[FloatTensor], src_mask: List[LongTensor], input_ids: LongTensor
    ) -> FloatTensor:
        assert len(src) == 1 and len(src_mask) == 1
        word_out, _ = self(src[0], src_mask[0], input_ids)
        return word_out


class PosDecoder(PosDecodeModel):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        dc: int,
        cross_coverage: bool,
        self_coverage: bool,
    ):
        super().__init__()
        self.pos_embed = nn.Sequential(
            nn.Linear(5,d_model),nn.GELU(),nn.LayerNorm(d_model)
        )  #[2b,l,5]  -->  [2b,l,256]
        self.pos_enc = WordPosEnc(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(d_model)

        self.model = _build_transformer_decoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
        )
        self.layernum_proj = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        ) 
        self.pos_proj = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        ) 
    def _build_attention_mask(self, length):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.full(
            (length, length), fill_value=1, dtype=torch.bool, device=self.device
        )
        mask.triu_(1)  # zero out the lower diagonal    

        return mask

    def forward(
        self, src: FloatTensor, src_mask: LongTensor, tgt: LongTensor,pos_tgt:FloatTensor
    ) -> Tuple[ FloatTensor,FloatTensor]:
        """generate output for tgt

        Parameters
        ----------
        src : FloatTensor
            [b, h, w, d]
        src_mask: LongTensor
            [b, h, w]
        tgt : LongTensor
            [b, l]
        pos_labels:LongTensor
            [b,l,5]
        is_test:bool
        
        Returns
        -------
        FloatTensor
            [b, l, vocab_size]
        """

        b , l = tgt.size()#16 6
        tgt_mask = self._build_attention_mask(l)
        tgt_pad_mask = tgt == vocab.PAD_IDX
        tgt_vocab=tgt
        pos_tgt=self.pos_embed(pos_tgt)  #[b,l,d]  
        pos_tgt = self.pos_enc(pos_tgt)  # [b, l, d]
        pos_tgt = self.norm(pos_tgt)


        h = src.shape[1]
        src = rearrange(src, "b h w d -> (h w) b d")
        src_mask = rearrange(src_mask, "b h w -> b (h w)")
        pos_tgt = rearrange(pos_tgt, "b l d -> l b d")

        out, attn = self.model(
            tgt=pos_tgt,
            memory=src,
            height=h,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_mask,
            tgt_vocab=tgt_vocab,
        )
        out_rearrange = rearrange(out, "l b d -> b l d")
        out_pos=self.pos_proj(out_rearrange)
        out_layernum=self.layernum_proj(out_rearrange)
        return out_layernum , out_pos, attn # attn b h l 16 6 6  16 6 5  out_layernum out_pos

    def transform(
        self, src: List[FloatTensor], src_mask: List[LongTensor], input_ids: LongTensor
    ) -> FloatTensor:
        assert len(src) == 1 and len(src_mask) == 1
        out_pos, _ = self(src[0], src_mask[0], input_ids,torch.zeros(1, dtype=torch.float, device=self.device))
        return out_pos
