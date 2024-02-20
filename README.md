# Learning Scalable Structural Representations for Link Prediction with Bloom Signatures [[paper](https://arxiv.org/abs/2312.16784)]

Existing GNN-based link prediction models improve the expressiveness of vanilla GNNs by capturing edge-wise structural features using labeling tricks or link prediction heuristics, which result in high computational overhead and limited scalability. To tackle this issue, we propose to learn structural link representations by augmenting the message-passing framework of GNNs with *Bloom signatures*. Bloom signatures are hashing-based compact encodings of node neighborhoods, which can be efficiently merged to recover various types of edge-wise structural features. We further show that any type of neighborhood overlap-based heuristic can be estimated by a neural network that takes Bloom signatures as input. GNNs with Bloom signatures are provably more expressive than vanilla GNNs and also more scalable than existing edge-wise models. Experimental results on five standard link prediction benchmarks show that our proposed model achieves comparable or better performance than existing edge-wise GNN models while being 3-200 $\times$ faster and more memory-efficient for online inference.

## Requirements
The codebase is tested with the following system configuration. Other configurations may work, but are untested.
1. Ubuntu 20.04
2. CUDA 11.8
3. Anaconda 4.11.0

## Installation

1. Create a fresh anaconda environment `bloom` and switch to it.
```
conda create -n bloom python=3.10
conda activate bloom
```
2. Install PyTorch 2.1 and the corresponding libraries.
```
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```
3. Install the dependencies listed in `requirements.txt`.
```
pip install -r requirements.txt
```

### [Optional] Set up W&B logging

We use Weights & Biases (W&B) to track experiment data. This is entirely optional. If you choose to log experiment data with W&B, after following the steps above, run the following command and paste your API key from your W&B portal.
```
wandb login
```

## Result Reproduction

To reproduce the results of our method reported in Table 3 of our paper, please use the following commands. Specifically, the argument `--dim_sign` controls the number of bits in the Bloom signature of each hop, e.g. `--dim_sign 4096 8192` allocates 2 hops of Bloom signatures for each node, where the first hop has 4096 bits and the second hop has 8192 bits.

1. ogbl-ddi:
```bash
python ogb_ddi.py --dim_sign 4096 8192 --use_sage --runs 5
```
2. ogbl-collab:
```bash
python ogb_collab.py --dim_sign 2048 8192 --use_valedges_as_input --epochs 500 --runs 5
```
3. ogbl-ppa:
```bash
python ogb_ppa.py --dim_sign 2048 8192 --epochs 750 --runs 5
```
4. ogbl-citation2:
```bash
python ogb_citation2.py --dim_sign 2048 8192 --runs 5
```
5. ogbl-vessel:
```bash
python ogb_vessel.py --dim_sign 2048 4096 --add_self_loops --use_sage --runs 5
```

## Citation

Please cite our paper if you find our work useful.
```
@inproceedings{zhang2024learning,
  author = {Zhang, Tianyi and Yin, Haoteng and Wei, Rongzhe and Li, Pan and Shrivastava, Anshumali},
  title = {Learning Scalable Structural Representations for Link Prediction with Bloom Signatures},
  year = {2024},
  url = {https://doi.org/10.1145/3589334.3645672},
  doi = {10.1145/3589334.3645672},
  booktitle = {Proceedings of the ACM Web Conference 2024},
  location = {Singapore, Singapore},
  series = {WWW '24}
}
```
