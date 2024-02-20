import argparse
from sklearn.utils import murmurhash3_32
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import is_undirected, to_undirected
import torch_geometric.transforms as T
from ogb.linkproppred import PygLinkPropPredDataset
import torch
import dgl.sparse as dglsp
from numba import njit, prange
from time import time

from numba import types
from numba.core.errors import TypingError
from numba.extending import overload


@overload(np.clip)
def impl_clip(a, a_min, a_max):
    # Source: https://jcristharif.com/numba-overload.html
    # Check that `a_min` and `a_max` are scalars, and at most one of them is None.
    if not isinstance(a_min, (types.Integer, types.Float, types.NoneType)):
        raise TypingError("a_min must be a_min scalar int/float")
    if not isinstance(a_max, (types.Integer, types.Float, types.NoneType)):
        raise TypingError("a_max must be a_min scalar int/float")
    if isinstance(a_min, types.NoneType) and isinstance(a_max, types.NoneType):
        raise TypingError("a_min and a_max can't both be None")

    if isinstance(a, (types.Integer, types.Float)):
        # `a` is a scalar with a valid type
        if isinstance(a_min, types.NoneType):
            # `a_min` is None
            def impl(a, a_min, a_max):
                return min(a, a_max)
        elif isinstance(a_max, types.NoneType):
            # `a_max` is None
            def impl(a, a_min, a_max):
                return max(a, a_min)
        else:
            # neither `a_min` or `a_max` are None
            def impl(a, a_min, a_max):
                return min(max(a, a_min), a_max)
    elif (
        isinstance(a, types.Array) and
        a.ndim == 1 and
        isinstance(a.dtype, (types.Integer, types.Float))
    ):
        # `a` is a 1D array of the proper type
        def impl(a, a_min, a_max):
            # Allocate an output array using standard numpy functions
            out = np.empty_like(a)
            # Iterate over `a`, calling `np.clip` on every element
            for i in range(a.size):
                # This will dispatch to the proper scalar implementation (as
                # defined above) at *compile time*. There should have no
                # overhead at runtime.
                out[i] = np.clip(a[i], a_min, a_max)
            return out
    else:
        raise TypingError(
            "`a` must be an int/float or a 1D array of ints/floats")

    # The call to `np.clip` has arguments with valid types, return our
    # numba-compatible implementation
    return impl


@njit('int32[::1](uint8[:,::1], int_)')
def bitSum(packed, axis=0):
    # Source: https://stackoverflow.com/questions/70659327/fast-bitwise-sum-in-python
    # sum func for bit-packed array
    n = packed.shape[0]
    m = packed.shape[1]*8
    if axis == 0:
        res = np.zeros(m, dtype=np.int32)
        for i in range(n):
            for j in range(m):
                res[j] += bool(packed[i, j//8] & (128 >> (j % 8)))
    elif axis == 1:
        res = np.zeros(n, dtype=np.int32)
        for i in range(n):
            for j in range(m):
                res[i] += bool(packed[i, j//8] & (128 >> (j % 8)))
    else:
        raise NotImplementedError
    return res


@njit(parallel=True)
def parallel_sketch(features, query, signatures, batch_size, dim_sign, rho, rlog, delta):
    # parallel version of BinSketch Alg 1
    num_query = query.shape[1]
    indices = [*range(0, num_query, batch_size)]
    for i in prange(len(indices)):
        bs = indices[i]
        be = min(bs + batch_size, num_query)
        u, v = query[:, bs:be]
        for j, dim in enumerate(dim_sign):
            sign_u, sign_v = signatures[j][u], signatures[j][v]
            # sign_uv =  signatures[j][query[:, bs:be]]
            nsa = np.clip(bitSum(sign_u, 1), None, dim-1)
            nsb = np.clip(bitSum(sign_v, 1), None, dim-1)
            # nsa, nsb = np.clip(sign_u.sum(-1), None, dim-1), np.clip(sign_v.sum(-1), None, dim-1)
            # nsab = sign_uv.sum(-1).clip(max=dim-1)
            na, nb = np.log(1 - nsa / dim) * \
                rlog[j], np.log(1 - nsb / dim) * rlog[j]
            # nab = (sign_u & sign_v).sum(-1)
            nab = bitSum(np.bitwise_and(sign_u, sign_v), 1)
            corr = rho[j] ** na + rho[j] ** nb + nab / dim - 1
            for k in np.arange(len(corr)):
                if corr[k] < delta:
                    corr[k] = 1-max(nsa[k], nsb[k])/dim
            features[bs:be, j] = na + nb - rlog[j] * np.log(corr)


class BloomSignature:
    def __init__(self, edge_index, num_nodes, dims, args, packing=True, parallel=False, debug=False):
        # node-wise feature / signature
        self.batch_size = args.batch_size
        self.dim_sign = np.array(dims)
        self.device = args.device
        self.cached_features = dict()
        self.debug = debug
        self.rho = 1 - 1 / self.dim_sign
        self.rlog = 1 / np.log(self.rho)
        self.delta = 1e-3
        self.parallel = parallel
        self.packing = packing

        if self.debug:
            return

        name = f'{args.dataset}_dim_{"_".join(map(str, self.dim_sign))}_seed_{args.seed}_hop_'

        if not is_undirected(edge_index):
            edge_index = to_undirected(edge_index)
            print('Converted to undirected edge index for BloomSignature.')

        # weight = np.ones(edge_index.shape[1], dtype=np.bool_)
        # mat = csr_matrix((weight, (edge_index[0].numpy(), edge_index[1].numpy())),
        #                        shape=(num_nodes, num_nodes))

        mat = dglsp.spmatrix(edge_index)
        adjs = []
        self.signatures = []

        time_start = time()
        for dim in self.dim_sign:
            adjs.append(mat@adjs[-1]) if len(adjs) > 0 else adjs.append(mat)
            signature = np.zeros((num_nodes, dim), dtype=np.bool_)
            hashing = murmurhash3_32(
                adjs[-1].col.numpy().astype(np.int32), positive=True).astype(np.int64)
            signature[adjs[-1].row.numpy(), hashing % dim] = True
            if self.packing:
                self.signatures.append(np.packbits(signature, axis=-1))
            else:
                self.signatures.append(signature)
        print(f'Finish BloomSignature for #{num_nodes} nodes in {time() - time_start:.2f}s')
        adjs = None

    def clean_cache(self):
        self.cached_features = dict()

    def get_pairwise_feature(self, query, name=None, parallel=True, filter=True, noshow=False):
        if self.debug:
            return torch.zeros(query.shape[1], 1, dtype=torch.float32)
        if isinstance(name, str) and name in self.cached_features:
            return self.cached_features[name]

        num_query = query.shape[1]
        hops = len(self.dim_sign)
        batch_size = self.batch_size

        query = query.cpu().numpy().astype(np.int32)
        features = np.zeros((num_query, hops), dtype=np.float32)

        start_time = time()
        if self.parallel and parallel:
            parallel_sketch(features, query, self.signatures, self.batch_size,
                            self.dim_sign, self.rho, self.rlog, self.delta)
            print(f'Finish parallel sketch of #{num_query} edges in {time() - start_time:.2f}s')
        else:
            for bs in tqdm(range(0, num_query, batch_size), disable=noshow):
                be = min(bs + batch_size, num_query)
                for i, dim in enumerate(self.dim_sign):
                    if self.packing:
                        sign_uv = np.unpackbits(
                            self.signatures[i][query[:, bs:be]], axis=-1)
                    else:
                        sign_uv = self.signatures[i][query[:, bs:be]]
                    # nsab = np.stack([bitSum(sign_uv[0], 1).clip(max=dim-1),
                    #                  bitSum(sign_uv[0], 1).clip(max=dim-1)])
                    # nab = bitSum(np.bitwise_and(sign_uv[0], sign_uv[1]), 1)
                    nsab = sign_uv.sum(-1).clip(max=dim-1)
                    na, nb = np.log(1 - nsab / dim) * self.rlog[i]
                    nab = (sign_uv[0] & sign_uv[1]).sum(-1)
                    corr = self.rho[i] ** na + \
                        self.rho[i] ** nb + nab / dim - 1
                    mask = corr < self.delta
                    corr[mask] = 1-nsab[:,mask].max(axis=0)/dim
                    features[bs:be, i] = na + nb - self.rlog[i] * np.log(corr)
                    # features[bs:be, i] = na + nb - self.rlog[i] * np.log(np.max([self.rho[i] ** na + self.rho[i] ** nb + nab / dim - 1, 1-nsab.max(axis=0)/dim], axis=0))
            if not noshow:
                print(f'Finish sequential sketch of #{num_query} edges in {time() - start_time:.2f}s')
        edge_feature = torch.from_numpy(features.clip(min=0)).to(self.device) if filter else torch.from_numpy(features).to(self.device)

        if isinstance(name, str):
            self.cached_features[name] = edge_feature
        return edge_feature


class BloomSketch:
    def __init__(self, data, edge_index, num_nodes, device, args):
        self.batch_size = args.hashing_batch_size
        self.bf_dim = args.bf_dim
        self.named_edge_features = dict()
        self.device = device
        self.debug = args.debug
        self.use_complement = args.use_complement
        self.use_containment = args.use_containment
        self.use_cosine = args.use_cosine
        self.use_cross_intersection = args.use_cross_intersection

        if self.debug:
            return

        if isinstance(args.degree_limit, int):
            loader = NeighborLoader(
                data,
                num_neighbors=[args.degree_limit],
                batch_size=data.num_nodes,
                shuffle=False,
            )
            edge_index = next(iter(loader)).edge_index

        bf_dim = args.bf_dim
        hops = len(bf_dim)
        for i in range(2, hops):
            if bf_dim[i] != bf_dim[i-1]:
                print("Bloom filters dimensions, except the first one, have to be the same.")
                exit(1)
        if not is_undirected(edge_index):
            edge_index = to_undirected(edge_index)
            print('Converted to undirected edge index for BloomSketch.')

        u, v = edge_index
        np_v = v.cpu().numpy().astype(np.int32)
        time_start = time()
        bloom_filters = torch.zeros(num_nodes, bf_dim[-1], dtype=torch.uint8)
        hash_codes = torch.from_numpy(murmurhash3_32(np_v, positive=True).astype(np.int64))
        bloom_filters[u, hash_codes % bf_dim[-1]] = 1
        if bf_dim[0] == bf_dim[-1]:
            self.bloom_filters = bloom_filters
        else:
            folded_bloom_filters = torch.zeros(num_nodes, bf_dim[0], dtype=torch.uint8)
            folded_bloom_filters[u, hash_codes % bf_dim[0]] = 1
            self.bloom_filters = folded_bloom_filters

        if args.packing:
            pos = torch.arange(1, 101, 1, dtype=torch.float32) / 100
            qt = torch.quantile(bloom_filters.sum(dim=1).to(torch.float32), pos).tolist()

            # for q in qt:
            #     wandb.log({
            #         'quantile_1hop': q,
            #     })
            print(qt)

        if hops >= 2:
            bloom_filters_2hop = torch.zeros(num_nodes, bf_dim[-1], dtype=torch.uint8)
            lst_u, lst_v = u.tolist(), v.tolist()
            for iu, iv in tqdm(zip(lst_u, lst_v), total=len(lst_u)):
                bloom_filters_2hop[iu] |= bloom_filters[iv]
            self.bloom_filters_2hop = bloom_filters_2hop
            
            if args.packing:
                pos = torch.arange(1, 101, 1, dtype=torch.float32) / 100
                qt = torch.quantile(bloom_filters_2hop.sum(dim=1).to(torch.float32), pos).tolist()

                # for q in qt:
                #     wandb.log({
                #         'quantile_2hop': q,
                #     })
                print(qt)

        if hops >= 3:
            bloom_filters_3hop = torch.zeros(num_nodes, bf_dim[-1], dtype=torch.uint8)
            lst_u, lst_v = u.tolist(), v.tolist()
            for iu, iv in tqdm(zip(lst_u, lst_v), total=len(lst_u)):
                bloom_filters_3hop[iu] |= bloom_filters_2hop[iv]
            self.bloom_filters_3hop = bloom_filters_3hop

            if args.packing:
                pos = torch.arange(1, 101, 1, dtype=torch.float32) / 100
                qt = torch.quantile(bloom_filters_3hop.sum(dim=1).to(torch.float32), pos).tolist()

                # for q in qt:
                #     wandb.log({
                #         'quantile_2hop': q,
                #     })
                print(qt)

        if hops >= 4:
            bloom_filters_4hop = torch.zeros(num_nodes, bf_dim[-1], dtype=torch.uint8)
            lst_u, lst_v = u.tolist(), v.tolist()
            for iu, iv in tqdm(zip(lst_u, lst_v), total=len(lst_u)):
                bloom_filters_4hop[iu] |= bloom_filters_3hop[iv]
            self.bloom_filters_4hop = bloom_filters_4hop

            if args.packing:
                pos = torch.arange(1, 101, 1, dtype=torch.float32) / 100
                qt = torch.quantile(bloom_filters_4hop.sum(dim=1).to(torch.float32), pos).tolist()

                # for q in qt:
                #     wandb.log({
                #         'quantile_2hop': q,
                #     })
                print(qt)
        print(f'Finish BloomSketch in {time() - time_start:.2f}s')

    @torch.no_grad()
    def get_edge_features(self, edge_index, name=None, noshow=False):
        if self.debug:
            return torch.zeros(edge_index.shape[1], 1, dtype=torch.float32)

        if isinstance(name, str) and name in self.named_edge_features:
            return self.named_edge_features[name]

        num_edges = edge_index.shape[1]
        bf_dim = self.bf_dim
        hops = len(bf_dim)
        batch_size = self.batch_size

        u, v = edge_index.cpu()
        its, complement_u, complement_v, containment_u, containment_v = [], [], [], [], []
        its_22, its_12, its_21, its_13, its_31, its_23, its_32, its_33, its_44 = [], [], [], [], [], [], [], [], []
        cosine11, cosine22, cosine33, cosine44 = [], [], [], []
        containment2_u, containment2_v = [], []
        containment3_u, containment3_v = [], []
        containment4_u, containment4_v = [], []
        n = 1 - 1 / np.array(bf_dim)
        log_n = np.log(n)
        full_bf_dim_sub1 = torch.full((batch_size, ), bf_dim[0] - 1).to(self.device)
        full_epsilon = torch.full((batch_size, ), 1e-6).to(self.device)
        for bs in tqdm(range(0, num_edges, batch_size), total=num_edges//batch_size+1, disable=noshow):
            be = min(bs + batch_size, num_edges)
            if be - bs != batch_size:
                full_bf_dim_sub1 = torch.full((be - bs, ), bf_dim[0] - 1).to(self.device)
                full_epsilon = torch.full((be - bs, ), 1e-6).to(self.device)
            full_bf_dim_sub1.fill_(bf_dim[0] - 1)
            bf_u, bf_v = self.bloom_filters[u[bs:be]].to(self.device), self.bloom_filters[v[bs:be]].to(self.device)
            na, nb, nab = torch.log(1 - torch.minimum(bf_u.sum(1), full_bf_dim_sub1) / bf_dim[0]) / log_n[0], \
                torch.log(1 - torch.minimum(bf_v.sum(1), full_bf_dim_sub1) / bf_dim[0]) / log_n[0], \
                torch.bitwise_and(bf_u, bf_v).sum(1)
            its.append(na + nb - 1 / log_n[0] * torch.log(n[0] ** na + n[0] ** nb + nab / bf_dim[0] - 1))
            mask = ~its[-1].isfinite()
            its[-1][mask] = torch.fmin(na[mask], nb[mask])
            if self.use_complement:
                complement_u.append(na - its[-1])
                complement_v.append(nb - its[-1])
            elif self.use_containment:
                containment_u.append(its[-1] / torch.maximum(na, full_epsilon))
                containment_v.append(its[-1] / torch.maximum(nb, full_epsilon))
            if self.use_cosine:
                cosine11.append(its[-1] / torch.sqrt(torch.maximum(na * nb, full_epsilon)))

            if hops >= 2:
                full_bf_dim_sub1.fill_(bf_dim[-1] - 1)
                bf2_u, bf2_v = self.bloom_filters_2hop[u[bs:be]].to(self.device), self.bloom_filters_2hop[v[bs:be]].to(self.device)
                na2, nb2, nab22 = torch.log(1 - torch.minimum(bf2_u.sum(1), full_bf_dim_sub1) / bf_dim[1]) / log_n[1], \
                    torch.log(1 - torch.minimum(bf2_v.sum(1), full_bf_dim_sub1) / bf_dim[1]) / log_n[1], \
                    torch.bitwise_and(bf2_u, bf2_v).sum(1)
                its_22.append(na2 + nb2 - 1 / log_n[1] * torch.log(n[1] ** na2 + n[1] ** nb2 + nab22 / bf_dim[1] - 1))
                mask = ~its_22[-1].isfinite()
                its_22[-1][mask] = torch.fmin(na2[mask], nb2[mask])

                # if args.use_cross_intersection:
                #     nab12 = torch.bitwise_and(bf_u, bf2_v).sum(1)
                #     its_12.append(na + nb2 - 1 / log_n * torch.log(n ** na + n ** nb2 + nab12 / bf_dim - 1))
                #     mask = ~its_12[-1].isfinite()
                #     its_12[-1][mask] = torch.fmin(na[mask], nb2[mask])

                #     nab21 = torch.bitwise_and(bf2_u, bf_v).sum(1)
                #     its_21.append(na2 + nb - 1 / log_n * torch.log(n ** na2 + n ** nb + nab21 / bf_dim - 1))
                #     mask = ~its_21[-1].isfinite()
                #     its_21[-1][mask] = torch.fmin(na2[mask], nb[mask])

                if self.use_containment:
                    containment2_u.append(its_22[-1] / torch.maximum(na2, full_epsilon))
                    containment2_v.append(its_22[-1] / torch.maximum(nb2, full_epsilon))

                if self.use_cosine:
                    cosine22.append(its_22[-1] / torch.sqrt(torch.maximum(na2 * nb2, full_epsilon)))

            if hops >= 3:
                bf3_u, bf3_v = self.bloom_filters_3hop[u[bs:be]].to(self.device), self.bloom_filters_3hop[v[bs:be]].to(self.device)
                na3, nb3, nab33 = torch.log(1 - torch.minimum(bf3_u.sum(1), full_bf_dim_sub1) / bf_dim[2]) / log_n[2], \
                    torch.log(1 - torch.minimum(bf3_v.sum(1), full_bf_dim_sub1) / bf_dim[2]) / log_n[2], \
                    torch.bitwise_and(bf3_u, bf3_v).sum(1)
                its_33.append(na3 + nb3 - 1 / log_n[2] * torch.log(n[2] ** na3 + n[2] ** nb3 + nab33 / bf_dim[2] - 1))
                mask = ~its_33[-1].isfinite()
                its_33[-1][mask] = torch.fmin(na3[mask], nb3[mask])

                # if args.use_cross_intersection:
                #     nab13 = torch.bitwise_and(bf_u, bf3_v).sum(1)
                #     its_13.append(na + nb3 - 1 / log_n * torch.log(n ** na + n ** nb3 + nab13 / bf_dim - 1))
                #     mask = ~its_13[-1].isfinite()
                #     its_13[-1][mask] = torch.fmin(na[mask], nb3[mask])

                #     nab31 = torch.bitwise_and(bf3_u, bf_v).sum(1)
                #     its_31.append(na3 + nb - 1 / log_n * torch.log(n ** na3 + n ** nb + nab31 / bf_dim - 1))
                #     mask = ~its_31[-1].isfinite()
                #     its_31[-1][mask] = torch.fmin(na3[mask], nb[mask])

                #     nab23 = torch.bitwise_and(bf2_u, bf3_v).sum(1)
                #     its_23.append(na2 + nb3 - 1 / log_n * torch.log(n ** na2 + n ** nb3 + nab23 / bf_dim - 1))
                #     mask = ~its_23[-1].isfinite()
                #     its_23[-1][mask] = torch.fmin(na2[mask], nb3[mask])

                #     nab32 = torch.bitwise_and(bf3_u, bf2_v).sum(1)
                #     its_32.append(na3 + nb2 - 1 / log_n * torch.log(n ** na3 + n ** nb2 + nab32 / bf_dim - 1))
                #     mask = ~its_32[-1].isfinite()
                #     its_32[-1][mask] = torch.fmin(na3[mask], nb2[mask])

                if self.use_containment:
                    containment3_u.append(its_33[-1] / torch.maximum(na3, full_epsilon))
                    containment3_v.append(its_33[-1] / torch.maximum(nb3, full_epsilon))

                if self.use_cosine:
                    cosine33.append(its_33[-1] / torch.sqrt(torch.maximum(na3 * nb3, full_epsilon)))

            if hops >= 4:
                bf4_u, bf4_v = self.bloom_filters_4hop[u[bs:be]].to(self.device), self.bloom_filters_4hop[v[bs:be]].to(self.device)
                na4, nb4, nab44 = torch.log(1 - torch.minimum(bf4_u.sum(1), full_bf_dim_sub1) / bf_dim[3]) / log_n[3], \
                    torch.log(1 - torch.minimum(bf4_v.sum(1), full_bf_dim_sub1) / bf_dim[3]) / log_n[3], \
                    torch.bitwise_and(bf4_u, bf4_v).sum(1)
                its_44.append(na4 + nb4 - 1 / log_n[3] * torch.log(n[3] ** na4 + n[3] ** nb4 + nab44 / bf_dim[3] - 1))
                mask = ~its_44[-1].isfinite()
                its_44[-1][mask] = torch.fmin(na4[mask], nb4[mask])

                if self.use_containment:
                    containment4_u.append(its_44[-1] / torch.maximum(na4, full_epsilon))
                    containment4_v.append(its_44[-1] / torch.maximum(nb4, full_epsilon))

                if self.use_cosine:
                    cosine44.append(its_44[-1] / torch.sqrt(torch.maximum(na4 * nb4, full_epsilon)))

        stacks = [torch.cat(its)]
        if self.use_complement:
            stacks.append(torch.cat(complement_u))
            stacks.append(torch.cat(complement_v))
        if self.use_containment:
            stacks.append(torch.cat(containment_u))
            stacks.append(torch.cat(containment_v))
        if self.use_cosine:
            stacks.append(torch.cat(cosine11))

        if hops >= 2:
            stacks.append(torch.cat(its_22))
            if self.use_cross_intersection:
                stacks.append(torch.cat(its_12))
                stacks.append(torch.cat(its_21))
            if self.use_containment:
                stacks.append(torch.cat(containment2_u))
                stacks.append(torch.cat(containment2_v))
            if self.use_cosine:
                stacks.append(torch.cat(cosine22))

        if hops >= 3:
            stacks.append(torch.cat(its_33))
            if self.use_cross_intersection:
                stacks.append(torch.cat(its_13))
                stacks.append(torch.cat(its_31))
                stacks.append(torch.cat(its_23))
                stacks.append(torch.cat(its_32))
            if self.use_containment:
                stacks.append(torch.cat(containment3_u))
                stacks.append(torch.cat(containment3_v))
            if self.use_cosine:
                stacks.append(torch.cat(cosine33))

        if hops >= 4:
            stacks.append(torch.cat(its_44))
            if self.use_containment:
                stacks.append(torch.cat(containment4_u))
                stacks.append(torch.cat(containment4_v))
            if self.use_cosine:
                stacks.append(torch.cat(cosine44))

        edge_feats = torch.stack(stacks).T
        if isinstance(name, str):
            self.named_edge_features[name] = edge_feats

        torch.cuda.empty_cache()
        return edge_feats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BloomSignature')
    parser.add_argument('--dataset', type=str, default='ogbl-citation2')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--batch_size', type=int, default=2 ** 16)
    parser.add_argument('--dim_sign', type=int,
                        nargs='+', default=[1024, 4096])
    parser.add_argument('--hashing_batch_size', type=int, default=2 ** 16)
    parser.add_argument('--bf_dim', type=int, nargs='+', default=[2048, 8192])
    parser.add_argument('--use-containment', action='store_true')
    parser.add_argument('--use-complement', action='store_true')
    parser.add_argument('--use-cosine', action='store_true')
    parser.add_argument('--use-cross-intersection', action='store_true')
    parser.add_argument('--degree-limit', type=int, default=None)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--packing', action='store_true')
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygLinkPropPredDataset(name=args.dataset,
                                     transform=T.ToSparseTensor(),
                                     root='./dataset/')
    data = dataset[0]
    split_edge = dataset.get_edge_split()
    # train_edge_index = torch.stack(
    #     [split_edge['train']['source_node'], split_edge['train']['target_node']])
    train_edge_index = split_edge['train']['edge'].t()

    bloom_sketch = BloomSignature(
        train_edge_index, data.num_nodes, args.dim_sign, args, parallel=True)
    # bloom_sketch = BloomSketch(data, train_edge_index, data.num_nodes, device, args)

    # source = split_edge['test']['source_node']
    # target = split_edge['test']['target_node']
    # target_neg = split_edge['test']['target_node_neg']
    # pos_edge_attr = bloom_sketch.get_pairwise_feature(torch.stack([source, target]))

    # source = source.view(-1, 1).repeat(1, 1000).view(-1)
    # target_neg = target_neg.view(-1)
    # neg_edge_attr = bloom_sketch.get_pairwise_feature(torch.stack([source, target_neg]))

    # feature = bloom_sketch.get_pairwise_feature(train_edge_index, name='train')
    pos_test_edge = split_edge['test']['edge'].t()
    neg_test_edge = split_edge['test']['edge_neg'].t()
    pos_edge_attr = bloom_sketch.get_pairwise_feature(pos_test_edge)
    neg_edge_attr = bloom_sketch.get_pairwise_feature(neg_test_edge)