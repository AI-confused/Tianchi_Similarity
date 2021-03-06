from transformers import BertTokenizer, AdamW, BertModel, BertPreTrainedModel
import torch.nn as nn
import torch
import numpy as np
from torch.nn import CrossEntropyLoss


class BertForSimilary(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSimilary, self).__init__(config)
        self.bert = BertModel(config)
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, labels=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            head_mask=head_mask)

#         sequence_output = outputs[0]
        pooled_output = outputs[1]
    
        # classification
        pooled_output = self.dropout(pooled_output)
        # 做 bert_enc mean
#         mask_2 = attention_mask # 其余等于 1 的部分，即有效的部分                
#         mask_2_expand = mask_2.unsqueeze_(-1).expand(pooled_output.size()).float()
#         sum_mask = mask_2_expand.sum(dim=1) # 有效的部分“长度”求和
#         sum_mask = torch.clamp(sum_mask, min=1e-9)
#         pooled_output = torch.sum(pooled_output * mask_2_expand, dim=1) / sum_mask

#         pooled_output = pooled_output.mean(dim=1)
        logits = self.classifier(pooled_output)
        
        if labels is not None:
#             loss_fct = CrossEntropyLoss()
            loss_fct = CrossEntropyLoss(weight=torch.from_numpy(np.array([0.5755,1])).float(), size_average=True).to('cuda')
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = loss
        else:
            outputs = nn.functional.softmax(logits,-1)

        return outputs
