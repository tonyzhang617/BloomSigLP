import argparse

import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.utils import to_undirected

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

from tqdm import tqdm

from logger import Logger

from sfeature import BloomSignature

import time
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, normalize=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, normalize=False))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, normalize=False))

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

    source_edge = split_edge['train']['source_node'].to(data.x.device)
    target_edge = split_edge['train']['target_node'].to(data.x.device)

    total_loss = total_examples = 0
    for perm in tqdm(DataLoader(range(source_edge.size(0)), batch_size, shuffle=True)):
        optimizer.zero_grad()

        h = model(data.x, data.adj_t)

        src, dst = source_edge[perm], target_edge[perm]

        pos_out = predictor(h[src], h[dst], data.edge_attr[perm])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        dst_neg = torch.randint(0, data.num_nodes, src.size(), dtype=torch.long, device=h.device)
        neg_edge_attr = bloom_sketch.get_pairwise_feature(torch.stack([src, dst_neg]).cpu(),  parallel=False,  noshow=True).to(h.device)
        neg_out = predictor(h[src], h[dst_neg], neg_edge_attr)
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, data, bloom_sketch, split_edge, evaluator, batch_size):
    predictor.eval()

    h = model(data.x, data.adj_t)

    def test_split(split, mode):
        source = split_edge[split]['source_node'].to(h.device)
        target = split_edge[split]['target_node'].to(h.device)
        target_neg = split_edge[split]['target_node_neg'].to(h.device)

        pos_preds = []
        pos_edge_attr = bloom_sketch.get_pairwise_feature(torch.stack([source, target]).cpu(), f'pos_{mode}')
        for perm in DataLoader(range(source.size(0)), batch_size):
            src, dst = source[perm], target[perm]
            pos_preds += [predictor(h[src], h[dst], pos_edge_attr[perm]).squeeze().cpu()]
        pos_pred = torch.cat(pos_preds, dim=0)

        neg_preds = []
        source = source.view(-1, 1).repeat(1, 1000).view(-1)
        target_neg = target_neg.view(-1)
        neg_edge_attr = bloom_sketch.get_pairwise_feature(torch.stack([source, target_neg]).cpu(), f'neg_{mode}')
        for perm in DataLoader(range(source.size(0)), batch_size):
            src, dst_neg = source[perm], target_neg[perm]
            neg_preds += [predictor(h[src], h[dst_neg], neg_edge_attr[perm]).squeeze().cpu()]
        neg_pred = torch.cat(neg_preds, dim=0).view(-1, 1000)

        return evaluator.eval({
            'y_pred_pos': pos_pred,
            'y_pred_neg': neg_pred,
        })['mrr_list'].mean().item()

    train_mrr = test_split('eval_train', 'eval_train')
    valid_mrr = test_split('valid', 'valid')
    sta = time.time()
    test_mrr = test_split('test', 'test')
    print(f'Inf time: {time.time()-sta}')

    return train_mrr, valid_mrr, test_mrr


def main():
    parser = argparse.ArgumentParser(description='OGBL-Citation2 (GNN)')
    parser.add_argument('--dataset', type=str, default='ogbl-citation2')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--use-containment', action='store_true')
    parser.add_argument('--use-complement', action='store_true')
    parser.add_argument('--use-cosine', action='store_true')
    parser.add_argument('--use-cross-intersection', action='store_true')
    parser.add_argument('--packing', action='store_true')
    parser.add_argument('--bf-dim', type=int, nargs='+', default=[1024, 4096])
    parser.add_argument('--dim_sign', type=int, nargs='+', default=[1024, 4096])
    parser.add_argument('--hashing-batch-size', type=int, default=2 ** 16)
    parser.add_argument('--degree-limit', type=int, default=None)
    parser.add_argument('--hops', type=int, default=1)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=2023)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygLinkPropPredDataset(name='ogbl-citation2',
                                     transform=T.ToSparseTensor(), root='./data/dataset/')
    data = dataset[0]
    
    data.adj_t = data.adj_t.to_symmetric()

    split_edge = dataset.get_edge_split()

    # We randomly pick some training samples that we want to evaluate on:
    torch.manual_seed(12345)
    idx = torch.randperm(split_edge['train']['source_node'].numel())[:86596]
    split_edge['eval_train'] = {
        'source_node': split_edge['train']['source_node'][idx],
        'target_node': split_edge['train']['target_node'][idx],
        'target_node_neg': split_edge['valid']['target_node_neg'],
    }

    train_edge_index = torch.stack([split_edge['train']['source_node'], split_edge['train']['target_node']])
    # node-wise features
    # bloom_sketch = BloomSketch(data, to_undirected(train_edge_index), data.num_nodes, device, args)
    bloom_sketch = BloomSignature(
        train_edge_index, data.num_nodes, args.dim_sign, args, parallel=True)
    # pre-cache training edge-wise features
    train_edge_attr = bloom_sketch.get_pairwise_feature(train_edge_index)
    edge_attr_dim = train_edge_attr.shape[1]
    print(f'Edge attribute dim = {edge_attr_dim}')
    data.edge_attr = train_edge_attr

    data = data.to(device)

    if args.use_sage:
        model = SAGE(data.num_features, args.hidden_channels,
                     args.hidden_channels, args.num_layers,
                     args.dropout).to(device)
    else:
        model = GCN(data.num_features, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout).to(device)

        # Pre-compute GCN normalization.
        adj_t = data.adj_t.set_diag()
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        data.adj_t = adj_t

    predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1, edge_attr_dim,
                              args.num_layers, args.dropout).to(device)

    evaluator = Evaluator(name='ogbl-citation2')
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(predictor.parameters()),
            lr=args.lr)

        for epoch in range(1, 1 + args.epochs):
            loss = train(model, predictor, data, bloom_sketch, split_edge, optimizer,
                         args.batch_size)
            print(f'Run: {run + 1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}')

            if epoch % args.eval_steps == 0:
                result = test(model, predictor, data, bloom_sketch, split_edge, evaluator,
                              args.batch_size)
                logger.add_result(run, result)

                if epoch % args.log_steps == 0:
                    train_mrr, valid_mrr, test_mrr = result
                    print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'Train: {train_mrr:.4f}, '
                          f'Valid: {valid_mrr:.4f}, '
                          f'Test: {test_mrr:.4f}')

        print('GraphSAGE' if args.use_sage else 'GCN')
        logger.print_statistics(run)

        bloom_sketch.clean_cache()
        
    print('GraphSAGE' if args.use_sage else 'GCN')
    logger.print_statistics()


if __name__ == "__main__":
    main()