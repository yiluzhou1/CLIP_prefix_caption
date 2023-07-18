import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union
from torch.utils.tensorboard import SummaryWriter


class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'


class ClipCocoDataset(Dataset):

    def __len__(self) -> int:
        return len(self.captions_tokens)

    def pad_tokens(self, item: int):
        tokens = self.captions_tokens[item]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            self.captions_tokens[item] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            self.captions_tokens[item] = tokens
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return tokens, mask

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        tokens, mask = self.pad_tokens(item)
        prefix = self.prefixes[self.caption2embedding[item]]
        if self.normalize_prefix:
            prefix = prefix.float()
            prefix = prefix / prefix.norm(2, -1)
        return tokens, mask, prefix

    def __init__(self, data_path: str,  prefix_length: int, gpt2_type: str = "gpt2",
                 normalize_prefix=False):
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        print("Data size is %0d" % len(all_data["clip_embedding"]))
        sys.stdout.flush()
        self.prefixes = all_data["clip_embedding"]
        captions_raw = all_data["captions"]
        self.image_ids = [caption["image_id"] for caption in captions_raw]
        self.captions = [caption['caption'] for caption in captions_raw]
        if os.path.isfile(f"{data_path[:-4]}_tokens.pkl"):
            with open(f"{data_path[:-4]}_tokens.pkl", 'rb') as f:
                self.captions_tokens, self.caption2embedding, self.max_seq_len = pickle.load(f)
        else:
            self.captions_tokens = []
            self.caption2embedding = []
            max_seq_len = 0
            for caption in captions_raw:
                self.captions_tokens.append(torch.tensor(self.tokenizer.encode(caption['caption']), dtype=torch.int64))
                self.caption2embedding.append(caption["clip_embedding"])
                max_seq_len = max(max_seq_len, self.captions_tokens[-1].shape[0])
            # self.max_seq_len = max_seq_len
            with open(f"{data_path[:-4]}_tokens.pkl", 'wb') as f:
                pickle.dump([self.captions_tokens, self.caption2embedding, max_seq_len], f)
        all_len = torch.tensor([len(self.captions_tokens[i]) for i in range(len(self))]).float()
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))


class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh, dropout=0.):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
                layers.append(nn.Dropout(dropout))  # adding dropout after activation
        self.model = nn.Sequential(*layers)
        if dropout != 0.:
            print(f'MLP dropout = {dropout}')


class MlpTransformer(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=nnf.relu, dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)
        if dropout != 0.:
            print(f'MlpTransformer dropout = {dropout}')

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class MultiHeadAttention(nn.Module):

    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)
        if dropout != 0.:
            print(f'MultiHeadAttention dropout = {dropout}')

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.dropout(out)  # Apply dropout here
        out = self.project(out)
        return out, attention


class TransformerLayer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        x_, attention = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x

    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=nnf.relu,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)
        if dropout != 0.:
            print(f'TransformerLayer dropout = {dropout}')


class Transformer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec: # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, mask)
            else:  # self or cross
                x = layer(x, y, mask)
        return x

    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dropout: float=0.,dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False):
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:  # cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, dropout=dropout, act=act, norm_layer=norm_layer))
            elif enc_dec:  # self
                layers.append(TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, dropout=dropout, act=act, norm_layer=norm_layer))
            else:  # self or cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, dropout=dropout, act=act, norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)
        if dropout != 0.:
            print(f'Transformer dropout = {dropout}')


class TransformerMapper(nn.Module):

    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]
        return out

    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 8, dropout: float=0.):
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(dim_embedding, 8, num_layers, dropout)
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)
        if dropout != 0.:
            print(f'TransformerMapper dropout = {dropout}')


class ClipCaptionModel(nn.Module):

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512,
                 num_layers: int = 8, mapping_type: MappingType = MappingType.MLP, dropout: float=0.):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.dropout = dropout
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if mapping_type == MappingType.MLP:
            self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2,
                                     self.gpt_embedding_size * prefix_length), dropout=dropout)
        else:
            self.clip_project = TransformerMapper(prefix_size, self.gpt_embedding_size, prefix_length,
                                                                     clip_length, num_layers, dropout=dropout)
        if dropout != 0.:
            print(f'ClipCaptionModel dropout = {dropout}')


class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        if self.dropout != 0.:
            print(f'ClipCaptionPrefix dropout = {self.dropout}')
        self.gpt.eval()
        return self

def default_method(obj):
    if isinstance(obj, MappingType):  # Assuming MappingType is your custom type
        # Provide your own method to convert obj into a JSON-serializable format
        return obj.value
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

def save_config(args: argparse.Namespace, output_dir: str):
    config = {}
    for key, item in args._get_kwargs():
        config[key] = item
    out_path = os.path.join(output_dir, f"{args.prefix}.json")
    with open(out_path, 'w') as outfile:
        json.dump(config, outfile, default=default_method)

def load_model(config_path: str, epoch_or_latest: Union[str, int] = '_latest'):
    with open(config_path) as f:
        config = json.load(f)
    parser = argparse.ArgumentParser()
    parser.set_defaults(**config)
    args = parser.parse_args()
    if type(epoch_or_latest) is int:
        epoch_or_latest = f"-{epoch_or_latest:03d}"
    model_path = os.path.join(args.out_dir, f"{args.prefix}{epoch_or_latest}.pt")
    if args.only_prefix:
        model = ClipCaptionPrefix(args.prefix_length)
    else:
        model = ClipCaptionModel(args.prefix_length)
    if os.path.isfile(model_path):
        print(f"loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        print(f"{model_path} is not exist")
    return model, parser

def get_next_folder_path(base_folder):
    i = 1
    while True:
        next_folder = os.path.join(base_folder, f'{i:03}')
        if not os.path.exists(next_folder):
            os.makedirs(next_folder)
            return next_folder
        i += 1

def count_parameters(model):
    """
    Count the number of parameters 
    """
    layer_count = 0
    for name, module in model.named_modules():
        layer_count += 1
        # print(f'Layer {layer_count}: {name} - {type(module)}')
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return layer_count, num_params

def train(train_dataset: ClipCocoDataset, model: ClipCaptionModel, args,
          warmup_steps: int = 5000, output_dir: str = ".", output_prefix: str = "",
          eval_dataset: Optional[ClipCocoDataset] = None):
    
    # Count model parameters
    layer_count, num_params = count_parameters(model)
    print(f'There are {layer_count} layers and {num_params} parameters in this {model.__class__.__name__} model.')
    
    device = torch.device('cuda:0')
    batch_size = args.bs
    epochs = args.epochs
    pretrained_weights_path = args.pretrained_weights_path
    weight_decay = args.weight_decay  # L2 regularization coefficient
    accumulation_steps = args.accumulation_steps
    lr = args.lr
    output_dir = get_next_folder_path(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = model.to(device)

    if pretrained_weights_path != '':
        # Load the pretrained weights
        try:
            model.load_state_dict(torch.load(pretrained_weights_path))
            print(f'The pretrained model {pretrained_weights_path} matched this model perfectly.')
        except:
            print(f'It looks the pretrained model {pretrained_weights_path} is different with this model. Trying to partially load the parameters...')
            # Load the state dict of the pre-trained model
            pretrained_dict = torch.load(pretrained_weights_path)
            # Get the state dict of the current model
            model_dict = model.state_dict()
            # Filter out unnecessary keys
            filtered_pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # Update the current model's dict
            model_dict.update(filtered_pretrained_dict)
            # Set the model's state dict
            model.load_state_dict(model_dict)
            print(f'The pretrained model {pretrained_weights_path} is partially loaded into this model successfully.')

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    if eval_dataset is not None:
        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    )
    save_config(args, output_dir)
    
    # create a SummaryWriter object
    writer = SummaryWriter(output_dir)
    # Initialize global step counter
    global_step = 0
    
    for epoch in range(epochs):
        # ======================== Training ================================
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        train_progress = tqdm(total=len(train_dataloader), desc=output_prefix)
        for idx, (tokens, mask, prefix) in enumerate(train_dataloader):
            model.zero_grad()
            tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
            outputs = model(tokens, prefix, mask)
            logits = outputs.logits[:, train_dataset.prefix_length - 1: -1]
            train_loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
            # normalize loss by gradient_accumulation_steps
            train_loss = train_loss / accumulation_steps
            train_loss.backward()
            
            # Perform optimizer step and scheduler step only after every gradient_accumulation_steps
            if ((idx + 1) % accumulation_steps == 0) or (idx + 1 == len(train_dataloader)):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            train_progress.set_postfix({"train_loss": train_loss.item()})
            train_progress.update()
            if (idx + 1) % 10000 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, f"{output_prefix}_latest.pt"),
                )
            # log the losses
            writer.add_scalar('train_loss', train_loss.item(), global_step)
            # Increment the global step
            global_step += 1

        if epoch % args.save_every == 0 or epoch == epochs - 1:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch:03d}.pt"),
            )

        train_progress.close()

        # ======================== Evaluating ==============================
        if eval_dataset is not None:
            print(f">>> Evaluating epoch {epoch}")
            eval_progress = tqdm(total=len(eval_dataloader), desc=output_prefix)
            total_eval_loss = 0
            model.eval()
            with torch.no_grad():
                for idx, (tokens, mask, prefix) in enumerate(eval_dataloader):
                    tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
                    outputs = model(tokens, prefix, mask)
                    logits = outputs.logits[:, eval_dataset.prefix_length - 1: -1]
                    eval_loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
                    total_eval_loss += eval_loss.item()
                    eval_progress.set_postfix({"eval_loss": eval_loss.item()})
                    eval_progress.update()

            # Calculate the average loss over all of the batches.
            avg_eval_loss = total_eval_loss / len(eval_dataloader)
            # Log the average eval loss after each epoch
            writer.add_scalar('eval_loss', avg_eval_loss, epoch)

            # Switch back to training mode
            model.train()        

    # close the writer
    writer.close()

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', default='./data/roco/train_ViT-B_32.pkl')
    parser.add_argument('--eval_data', default='') #'./data/roco/eval_ViT-B_32.pkl'
    parser.add_argument('--out_dir', default='./checkpoints')
    parser.add_argument('--prefix', default='roco_prefix', help='prefix for saved filenames')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--prefix_length_clip', type=int, default=10)
    parser.add_argument('--bs', type=int, default=40)
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true')
    parser.add_argument('--mapping_type', type=str, default='mlp', help='mlp/transformer')
    parser.add_argument('--num_layers', type=int, default=8)
    # parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'))
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    parser.add_argument('--pretrained_weights_path', type=str, default='')
    parser.add_argument('--dropout', type=float, default=0., help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=0., help='weight_decay rate')
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    
    args = parser.parse_args()
    prefix_length = args.prefix_length
    train_dataset = ClipCocoDataset(args.train_data, prefix_length, normalize_prefix=args.normalize_prefix)
    if args.eval_data != '':
        eval_dataset = ClipCocoDataset(args.eval_data, prefix_length, normalize_prefix=args.normalize_prefix)
    else:
        eval_dataset = None
    # prefix_dim = 640 if args.is_rn else 512
    if args.clip_model_type in ['RN50', 'RN50x4']:
        prefix_dim = 640
    elif args.clip_model_type in ['RN50x64']:
        prefix_dim = 1024
    elif args.clip_model_type in ['ViT-L/14', 'ViT-L/14@336px']:
        prefix_dim = 768
    else:
        prefix_dim = 512

    args.mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}[args.mapping_type]
    if args.only_prefix:
        model = ClipCaptionPrefix(prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=args.num_layers, mapping_type=args.mapping_type, dropout=args.dropout)
        print("Train only prefix")
    else:
        model = ClipCaptionModel(prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=args.num_layers, mapping_type=args.mapping_type, dropout=args.dropout)
        print("Train both prefix and GPT")
        sys.stdout.flush()
    train(train_dataset, model, args, output_dir=args.out_dir, output_prefix=args.prefix, eval_dataset=eval_dataset)


if __name__ == '__main__':
    main()
