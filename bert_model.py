import torch
import torch.nn as nn
from transformers import BertTokenizer, BertConfig
from transformers import BertModel, BertPreTrainedModel

bert_model_name = 'bert-base-chinese'
BERT_MODEL_PATH = f'./pretrained_models/{bert_model_name}/'
model_config = BertConfig.from_pretrained(bert_model_name)


class ChatBot(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                labels=None):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (
            start_logits,
            end_logits,
        ) + outputs[2:]

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)


class ChatModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.bert_model = BertModel.from_pretrained(BERT_MODEL_PATH,
                                                    config=model_config)

    def forward(self, tokens, segments):
        outputs = self.bert_model(tokens, token_type_ids=segments)
        print(type(outputs))
        print(outputs.shape)
        return outputs
