import logging

import torch
import torch.nn as nn
from adapters.common import AdapterConfig, freeze_all_parameters
from transformers import BertModel
from transformers.models.bert.modeling_bert import BertSelfOutput
logging.basicConfig(level=logging.INFO)

class Adapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = torch.nn.Linear(config.hidden_size, config.adapter_size)
        self.fc2 = torch.nn.Linear(config.adapter_size, config.hidden_size)
        self.activation = torch.nn.ReLU()

    def forward(self, x, add_residual=True):
        residual = x
        h = self.activation(self.fc1(x))
        h = self.activation(self.fc2(h))
        if add_residual:
            output = residual + h
        else:
            output = h
        return output

class AdaptedSelfOutput(nn.Module):
    def __init__(self,
                 self_output: BertSelfOutput,
                 config: AdapterConfig):
        super(AdaptedSelfOutput, self).__init__()
        self.self_output = self_output
        self.adapter = Adapter(config)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor):
        hidden_states = self.self_output.dense(hidden_states)
        hidden_states = self.self_output.dropout(hidden_states)
        hidden_states = self.adapter(hidden_states)
        hidden_states = self.self_output.LayerNorm(
            hidden_states + input_tensor)
        return hidden_states


def adapt_self_output(config: AdapterConfig):
    return lambda self_output: AdaptedSelfOutput(self_output, config=config)


def add_adapters(model: BertModel, config: AdapterConfig) -> BertModel:
    for layer in model.encoder.layer:
        layer.attention.output = adapt_self_output(
            config)(layer.attention.output)
        layer.output = adapt_self_output(config)(layer.output)
    return model


def unfreeze_adapters(model: nn.Module) -> nn.Module:
    # Unfreeze trainable parts — layer norms and adapters
    for name, sub_module in model.named_modules():
        if isinstance(sub_module, (Adapter, nn.LayerNorm)):
            for param_name, param in sub_module.named_parameters():
                param.requires_grad = True
    return model


def load_adapter_model(model: nn.Module, adapter_size: int = 512, checkpoint: str = None):
    #
    adapter_config = AdapterConfig(
        hidden_size=768,
        adapter_size=adapter_size,
        adapter_act='relu',
        adapter_initializer_range=1e-2
    )
    try:
        model.bert = add_adapters(model.bert, adapter_config)
        # freeze the bert model, unfreeze adapter
        model.bert = freeze_all_parameters(model.bert)
        model.bert = unfreeze_adapters(model.bert)
    except:
        model.roberta = add_adapters(model.roberta, adapter_config)
        model.roberta = freeze_all_parameters(model.roberta)
        model.roberta = unfreeze_adapters(model.roberta)

    if checkpoint is not None and checkpoint != 'None':
        print("loading checkpoint...")
        model_dict = model.state_dict()
        pretrained_dict = torch.load(checkpoint)
        # 过滤操作
        new_dict = {k: v for k, v in pretrained_dict.items()
                    if k in model_dict.keys()}
        model_dict.update(new_dict)
        # 打印出来，更新了多少的参数
        print('Total : {} params are loaded.'.format(len(pretrained_dict)))
        model.load_state_dict(model_dict)
        print("loaded finished!")
    else:
        print('No checkpoint is included')
    return model


def save_adapter_model(model: nn.Module, save_path: str, accelerator=None):
    model_dict = {k: v for k, v in model.state_dict().items()
                  if 'adapter' in k}
    if accelerator is not None:
        accelerator.save(model_dict, save_path)
    else:
        torch.save(model_dict, save_path)