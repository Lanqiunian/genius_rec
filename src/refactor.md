# GeniusRec模型重构指令

## 核心目标
将项目重构为一个高效且目标一致的序列推荐模型。该模型使用HSTU作为编码器，Transformer作为解码器，并采用“全量词汇表训练”（通过标准交叉熵损失实现）的策略，彻底废弃有问题的“Sampled Softmax”方法，以解决训练与评估目标不匹配导致的模型不收敛问题。

---

### **步骤一：调整核心训练策略 (`src/config.py`)**

1.  **定位文件**: `src/config.py`。
2.  **执行操作**:
    * 在`finetune`字典中，找到并**删除**或注释掉以下两行，因为我们将不再使用显式负采样：
        * `"use_sampled_softmax": True,`
        * `"num_neg_samples": 512,`
    * **【关键】** 将`finetune`字典中的`temperature`参数从`0.05`**修改为**一个更合理的值，建议为`0.2`。
        * `"temperature": 0.2`
    * **理由**: 废弃采样策略，并调整温度以防止模型在初期过分自信，促进学习更泛化的表示。

---

### **步骤二：确认数据管道 (`src/dataset.py`)**

1.  **定位文件**: `src/dataset.py`。
2.  **确认版本**: 确保当前使用的`Seq2SeqRecDataset`类是“最终修复版：随机分割的前缀预测后缀”版本。
3.  **验证逻辑**:
    * 该版本在训练时随机分割用户历史序列为`source_seq`和`target_seq`。
    * 该版本**不包含**任何生成`negative_samples`的逻辑。
    * `__getitem__`方法最终返回一个只包含`source_ids`, `decoder_input_ids`, `labels`三个键的字典。
4.  **执行操作**: 确认项目引用的为此正确版本，无需修改。

---

### **步骤三：修正解码器输出层 (`src/decoder.py`)**

1.  **定位文件**: `src/decoder.py`。
2.  **定位方法**: `GenerativeDecoder`类中的`forward`方法。
3.  **执行操作**:
    * 找到方法末尾计算`final_logits`的逻辑部分（注释为“步骤 3”）。
    * **完全删除**所有与`use_sampled`相关的`if/else`条件判断，包括`gather`正负样本logits的所有代码。
    * 修改为**无条件地**计算全量Logits，并将最终逻辑简化为：
        ```python
        # --- 步骤 3: 【最终的、内存安全的】输出层 ---
        # 首先，将最终的隐状态通过共享的嵌入层权重投影到整个词汇表的 logits 空间
        full_logits = F.linear(self.final_hidden_norm(final_hidden_state), item_embedding_layer.weight)

        # 在训练时应用温度参数，以平滑输出
        if self.training:
            temperature = self.finetune_config.get('temperature', 1.0) # 从配置读取温度
            full_logits = full_logits / temperature

        # 评估/推理模式下，直接使用原始logits
        final_logits = full_logits
        
        weights_to_return = expert_weights.squeeze(-1) if return_weights else None
        
        return final_logits, weights_to_return, balancing_loss, final_hidden_state
        ```
    * **理由**: 确保解码器始终输出对整个词汇表的预测，为后续的全量损失计算做准备。

---

### **步骤四：修复并完善训练与评估循环 (`train_GeniusRec.py`, `unified_evaluation.py`)**

这是最关键的一步，确保模型接收到正确的输入和掩码。

1.  **定位文件**: `src/train_GeniusRec.py`。
2.  **定位函数**: `train_one_epoch`。
3.  **执行操作**:
    * **移除负采样相关代码**: 删除从`batch`中获取`negative_samples`的所有代码。
    * **【关键修复】添加解码器掩码**: 在`optimizer.zero_grad()`之前，必须为解码器创建`target_padding_mask`。这是之前版本缺失的关键一步，对Transformer解码器的自注意力至关重要。
        ```python
        source_ids = batch['source_ids'].to(device)
        decoder_input_ids = batch['decoder_input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        source_padding_mask = (source_ids == pad_token_id)
        # ↓↓↓ 添加下面这行关键代码 ↓↓↓
        target_padding_mask = (decoder_input_ids == pad_token_id)
        ```
    * **修正模型调用**: 确保调用`model`时，传入了所有必需的参数，特别是`target_padding_mask`。
        ```python
        logits, gate_weights, balancing_loss, _ = model(
            source_ids=source_ids,
            decoder_input_ids=decoder_input_ids,
            source_padding_mask=source_padding_mask,
            target_padding_mask=target_padding_mask, # 确保传入
            item_embedding_layer=model.encoder.item_embedding, # 确保传入共享嵌入层
            return_weights=True
        )
        ```
    * **确认损失计算**: 确保损失计算部分是简洁、正确的全量计算版本。
        ```python
        task_loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        ```

4.  **定位文件**: `src/unified_evaluation.py`。
5.  **定位函数**: `evaluate_model_validation_with_ranking`。
6.  **执行操作**:
    * **【关键修复】** 与训练循环类似，此函数中也**必须**为解码器创建和传递`target_padding_mask`。
        ```python
        # 在函数内部的 for 循环中
        source_padding_mask = (source_ids == pad_token_id)
        # ↓↓↓ 添加下面这行关键代码 ↓↓↓
        target_padding_mask = (decoder_input_ids == pad_token_id)
        
        # ... 调用模型时确保传入 ...
        logits, gate_weights, _, _ = model(
            ...,
            target_padding_mask=target_padding_mask, # 确保传入
            ...
        )
        ```
    * **理由**: 确保训练和验证逻辑完全一致，特别是解码器部分的注意力掩码，避免因实现差异导致评估结果不准。

---

### **最终检查**
* 确保项目中所有对`model`的调用都遵循了更新后的参数列表。
* 删除所有不再使用的配置项和代码块，保持代码库整洁。
* 从一个干净的状态开始新的训练，不要加载旧的、由错误方法训练出的检查点。