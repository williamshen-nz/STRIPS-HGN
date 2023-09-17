# STRIPS-HGN
STRIPS-HGN is a framework for learning domain-independent planning heuristics completely from scratch using the hypergraph induced by the delete-relaxation of a STRIPS problem. 

For any issues please contact the authors of the paper. You can find our emails in the [paper](https://shen.nz/papers/shen-stripshgn-20.pdf).

**New (2023-09)** - fixed bug when using batch size > 1 and added [tips for making training more stable](#making-training-more-stable).

## Usage
### Directory Structure
- `benchmarks`: contains the PDDL domain and problems
- `experiments`: the experiments used in the paper as Python scripts
- `scripts`: helpful scripts used to run experiments on AWS. The `scripts/ec2_setup.sh` script could give hints to installing and running STRIPS-HGN
- `src`
  - `hypergraph_nets`: Hypergraph Networks implementation based off [Graph Networks](https://github.com/deepmind/graph_nets). Note this implementation is sub-optimal and can be improved.
  - `strips_hgn`: logic for setting up, running and collecting results for the experiments. Includes hypergraph generation, feature mapping, generating training data, etc.
- `tests`: some unit tests

### Dependencies
- You will need Python 3.6+
- Use pip to install the requirements in `requirements.txt` (`pip install -r requirements.txt`)
- Clone my custom version of Fast Downward (https://github.com/williamshen-nz/fast_downward.git) which you will need to build (use `build.py`)

I would recommend using a virtual environment. The entry point for training is `src/train.py` and for evaluation is `src/eval.py`. Use the help flag (`-h, --help`) to get usage information.

## Known Issues and TODOs
1. Move evaluation platform to Fast Downward (this is already completed and working in a separate repository)
2. Move remaining experiment scripts from old codebase
   - Hanoi
   - Matching Blocksworld
   - Sokoban
   - Domain-Independent Experiments
3. Optimise Hypergraph Networks

### Making Training more Stable
Training STRIPS-HGN can be quite unstable, which is partially why we used the *k* folds method during training. Here are
some tips to make training more stable:

- Remove the ReLU activation function in the output transform in `EncodeProcessDecode` (remove the `nn.ReLU(inplace=True)` from `edge_model`, `node_model`, and `global_model`).
    - When the STRIPS-HGN is initialized with weights such that the output is negative, the ReLU will zero the gradients and the network will not learn. Removing the ReLU and just 
      keeping the linear layer seems to solve this problem. 
    - Note that you will need to `torch.clamp(outputs, min=0)` or `min(0, outputs)`, so the outputs are non-negative and are hence valid heuristics.
- Using a smaller learning rate such as 1e-4 (the paper used 1e-3), or automatically reduce the learning rate when the loss plateaus
  (you can try `torch.optim.lr_scheduler.ReduceLROnPlateau`).

Thanks to Carlos Núñez Molina (University of Granada) and Dillon Chen (Australian National University) for these tips.
These changes mean you can try reducing the number of folds or potentially not use them at all.

### Past Updates
**17th September 2023**: Added tips for addressing unstable training and fixed bug that didn't merge `HypergraphsTuple` correctly by accumulating the node indices.
This meant that using batch size > 1 would not work properly. Thanks to Carlos Núñez Molina for raising these issues.

**Update - 19th April 2021**: apologies for the delay. Unfortunately, I have not had time to work on research so I am releasing the code-base as is. 

**Update - 21st Sept 2020**: the implementation is ready but just needs some final cleaning up and testing. Due to work and other circumstances I have been unable to spend much time on research. I hope to release update this repository by [ICAPS](https://icaps20.icaps-conference.org/) (October 19-30). For those who want a copy of the code in the meantime please email me. 

**Update - 21st June 2020**: our re-implementation of STRIPS-HGN is almost ready to go and is currently undergoing testing and experiments. We intend to make it available before the end of June. 

___

Please consider citing our paper if you found our work useful:
```
@inproceedings{shen20:stripshgn,
  author = {Shen, William and Trevizan, Felipe and Thi{\'e}baux, Sylvie},
  title = {Learning Domain-Independent Planning Heuristics with Hypergraph Networks},
  booktitle = {Proc. of 30th Int. Conf. on Automated Planning and Scheduling (ICAPS)},
  year = {2020},
}
```
