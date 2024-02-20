import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn import Parameter

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv
from torch_sparse import SparseTensor

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

from tqdm import tqdm
from logger import Logger

from sfeature import BloomSignature

from collections import defaultdict

import wandb

import time

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))

        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, attr_dim, num_layers,
                 dropout, beta=1.0, ln=True, twolayerlin=False, tailact=False):
        super(LinkPredictor, self).__init__()

        self.register_parameter("beta", Parameter(beta*torch.ones((1))))
        lnfn = lambda dim, ln: torch.nn.LayerNorm(dim) if ln else torch.nn.Identity()

        self.xcnlin = torch.nn.Sequential(
            torch.nn.Linear(attr_dim, hidden_channels),
            torch.nn.Dropout(dropout, inplace=True), torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), torch.nn.Dropout(dropout, inplace=True),
            torch.nn.ReLU(inplace=True), torch.nn.Linear(hidden_channels, hidden_channels) if not tailact else torch.nn.Identity())
        
        self.xijlin = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels), lnfn(hidden_channels, ln),
            torch.nn.Dropout(dropout, inplace=True), torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_channels, hidden_channels) if not tailact else torch.nn.Identity())

        self.lins = torch.nn.Sequential(torch.nn.Linear(hidden_channels, hidden_channels),
                                 lnfn(hidden_channels, ln),
                                 torch.nn.Dropout(dropout, inplace=True),
                                 torch.nn.ReLU(inplace=True),
                                 torch.nn.Linear(hidden_channels, hidden_channels) if twolayerlin else torch.nn.Identity(),
                                 lnfn(hidden_channels, ln) if twolayerlin else torch.nn.Identity(),
                                 torch.nn.Dropout(dropout, inplace=True) if twolayerlin else torch.nn.Identity(),
                                 torch.nn.ReLU(inplace=True) if twolayerlin else torch.nn.Identity(),
                                 torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for net in [self.xcnlin, self.xijlin, self.lins]:
            for layer in net:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

    def forward(self, x_i, x_j, edge_attr):
        xij = x_i * x_j
        xij = self.xijlin(xij)
        xcn = self.xcnlin(edge_attr)
        x = self.lins(xcn*self.beta+xij)
        return torch.sigmoid(x)



def train(model, predictor, data, bloom_sketch, split_edge, optimizer, batch_size):
    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(data.x.device)

    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size, shuffle=True):

        optimizer.zero_grad()

        h = model(data.x, data.adj_t)

        edge = pos_train_edge[perm].t()
        pos_out = predictor(h[edge[0]], h[edge[1]], data.edge_attr[perm])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        edge = torch.randint(0, data.num_nodes, edge.size(), dtype=torch.long,
                             device=h.device)
        
        neg_edge_attr = bloom_sketch.get_pairwise_feature(edge, parallel=False, noshow=True)
        neg_out = predictor(h[edge[0]], h[edge[1]], neg_edge_attr)
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, data, bloom_sketch, split_edge, evaluator, batch_size):
    model.eval()

    dt = 0
    sta = time.time()
    h = model(data.x, data.adj_t)
    dt += time.time()-sta

    pos_train_edge = split_edge['train']['edge'].to(h.device)
    pos_valid_edge = split_edge['valid']['edge'].to(h.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(h.device)
    pos_test_edge = split_edge['test']['edge'].to(h.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(h.device)

    pos_train_preds = []
    pos_edge_attr = bloom_sketch.get_pairwise_feature(pos_train_edge.t().cpu(), 'pos_train')
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(h[edge[0]], h[edge[1]], pos_edge_attr[perm]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    pos_edge_attr = bloom_sketch.get_pairwise_feature(pos_valid_edge.t().cpu(), 'pos_valid')
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(h[edge[0]], h[edge[1]], pos_edge_attr[perm]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    neg_edge_attr = bloom_sketch.get_pairwise_feature(neg_valid_edge.t().cpu(), 'neg_valid')
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h[edge[0]], h[edge[1]], neg_edge_attr[perm]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    sta = time.time()
    pos_test_preds = []
    pos_edge_attr = bloom_sketch.get_pairwise_feature(pos_test_edge.t().cpu(), 'pos_test', parallel=False)
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]], pos_edge_attr[perm]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    neg_edge_attr = bloom_sketch.get_pairwise_feature(neg_test_edge.t().cpu(), 'neg_test', parallel=False)
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]], neg_edge_attr[perm]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)
    dt += time.time() - sta
    wandb.log({'d_inf': dt})

    results = {}
    for K in [10, 50, 100]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)
    
    train_auc = evaluator._eval_rocauc(pos_train_pred, neg_valid_pred, 'torch')[f'rocauc']
    valid_auc = evaluator._eval_rocauc(pos_valid_pred, neg_valid_pred, 'torch')[f'rocauc']
    test_auc = evaluator._eval_rocauc(pos_test_pred, neg_test_pred, 'torch')[f'rocauc']
    results['AUC'] = (train_auc, valid_auc, test_auc)

    return results


def main():
    parser = argparse.ArgumentParser(description='OGBL-COLLAB (GNN)')
    parser.add_argument('--dataset', type=str, default='ogbl-collab')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_valedges_as_input', action='store_true')
    parser.add_argument('--use_node_embedding', action='store_true')
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--dim_sign', type=int, nargs='+', default=[2048, 8192])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=2023)
    args = parser.parse_args()
    print(args)

    wandb.init(
        project=f"bloom-link-prediction-{args.dataset}",
        config=vars(args),
    )


    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygLinkPropPredDataset(name=args.dataset)

    data = dataset[0]
    edge_index = data.edge_index
    data.edge_weight = data.edge_weight.view(-1).to(torch.float)
    data = T.ToSparseTensor()(data)

    if args.use_node_embedding:
        data.x = torch.cat([data.x, torch.load('embedding.pt')], dim=-1)

    split_edge = dataset.get_edge_split()

    if args.use_valedges_as_input:
        val_edge_index = split_edge['valid']['edge'].t()
        full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
        data.full_adj_t = SparseTensor.from_edge_index(full_edge_index).t()
        data.full_adj_t = data.full_adj_t.to_symmetric()
    else:
        data.full_adj_t = data.adj_t

    train_edge_index = full_edge_index if args.use_valedges_as_input else edge_index
    bloom_sketch = BloomSignature(train_edge_index, data.num_nodes, args.dim_sign, args, parallel=True)
    train_edge_attr = bloom_sketch.get_pairwise_feature(train_edge_index)
    edge_attr_dim = train_edge_attr.shape[1]
    print(f'Edge attribute dim = {edge_attr_dim}')

    data = data.to(device)
    data.edge_attr = train_edge_attr

    if args.use_sage:
        model = SAGE(data.num_features, args.hidden_channels,
                     args.hidden_channels, args.num_layers,
                     args.dropout).to(device)
    else:
        model = GCN(data.num_features, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout).to(device)

        # # Pre-compute GCN normalization.
        # adj_t = data.adj_t.set_diag()
        # deg = adj_t.sum(dim=1).to(torch.float)
        # deg_inv_sqrt = deg.pow(-0.5)
        # deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        # adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        # data.adj_t = adj_t

    predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1, edge_attr_dim,
                              args.num_layers, args.dropout).to(device)

    evaluator = Evaluator(name=args.dataset)
    loggers = {
        'Hits@10': Logger(args.runs, args),
        'Hits@50': Logger(args.runs, args),
        'Hits@100': Logger(args.runs, args),
        'AUC': Logger(args.runs, args),
    }

    for run in range(args.runs):
        best_results = defaultdict(lambda: (0, 0))
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(predictor.parameters()),
            lr=args.lr)

        for epoch in range(1, 1 + args.epochs):
            sta = time.time()
            loss = train(model, predictor, data, bloom_sketch, split_edge, optimizer,
                         args.batch_size)
            wandb.log({"run": run+1, "epoch": epoch, "loss": loss, "d_train": time.time()-sta,})

            if epoch % args.eval_steps == 0:
                results = test(model, predictor, data, bloom_sketch, split_edge, evaluator, args.batch_size)
                for key, result in results.items():
                    loggers[key].add_result(run, result)

                if epoch % args.log_steps == 0:
                    print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, ')
                    for key, result in results.items():
                        train_hits, valid_hits, test_hits = result
                        if (valid_hits > best_results[key][0]):
                            best_results[key] = (valid_hits, test_hits)
                        print(key,
                              f'Train: {100 * train_hits:.2f}%, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')
                        wandb.log({
                            f"train_{key}": 100 * train_hits,
                            f"valid_{key}": 100 * valid_hits,
                            f"test_{key}": 100 * test_hits,
                            f"best_test_{key}": 100 * best_results[key][1],
                        })

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)

        bloom_sketch.clean_cache()

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()


if __name__ == "__main__":
    main()