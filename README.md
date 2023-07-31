# Simple Molucular Dynamics Surrogate Model with Graph Nets

This repository contains a demonstration of learning to simulate a physics process such as molecular dynamics with graph neural networks (GNNs). The model is inspired by and an implementation of a culmination of work out of a group in Deepmind. Their graph_nets library is the core data structure that is used throughout this codebase. A few good resources from their group are:
- [Relational inductive biases, deep learning, and graph networks](https://arxiv.org/abs/1806.01261)
  - [graph_nets](https://github.com/deepmind/graph_nets)
- [Learning to Simulate Complex Physics with Graph Networks](https://arxiv.org/abs/2002.09405)
  - [learning_to_simulate](https://github.com/deepmind/deepmind-research/tree/master/learning_to_simulate)
- [Interaction Networks for Learning about Objects, Relations and Physics](https://arxiv.org/abs/1612.00222)


## Usage: Train a model and display trajectories
![2 atom rollout](./2_atoms_demo.gif)

This library uses the Tensorflow 2 API and is not compatible with Tensorflow 1. The graph_nets and sonnet versions must be compatible with Tensorflow 2. *TODO: more detailed requirements and compatibility* 

### Training

The `yaml` and `argparse` libraries are used to configure the input deck for training. The default configuration file is `config.yaml`. Each of the configuartion parameters are stored as keys in an `argparse.ArgumentParser` like object, which can be modifed at the command line or in a separate `yaml` file. For example simply running:
```
python train_md.py
```
Will take all the default options set in `config.yaml`. If you wish to modify any of these parameters it can be done with any of the following:
1. modify at the command line:
```
python train_md.py --num_time_steps 500 --cutoff 5
```
2. create a separate `yaml` file that will overwrite the set parameters, and keep the rest default. For example, if you only wish to modify `num_time_steps` and `cutoff` and keep the rest default, create a file `config2.yaml`:
```yaml
num_time_steps: 500
cutoff: 5
```
Then run:
```
python --config config2.yaml
```
3. Use both a separate `yaml` file and command line options. For example, if you wish to modify `num_time_steps` and `cutoff` in the `yaml` but `step_size` in the command line, you could use the previous `config2.yaml` file and run:
```
python --config config2.yaml --step_size 0.5
```
Note: The command line options are the final say and will overwrite shared parameters in `yaml`. 

---

During training, the following are printed to the terminal:
- iteration number
- elapsed time
- 1-step loss on training set
- Rollout loss on test-set
  
After training, the model's weights will be saved in a directory set by `args.model_save_direcotry` (default is `./models`) which can be later be accessed by the following:
```
model = learned_graph_simulator.LearnedSimulator(...)  # where ... are the required parameters to define the model
weights_path = 'path_to_model_weights'  # this path is printed to the terminal at the end of training
utils_md.load_weights(model, weights_path)
```

### Visualization
*TODO: Notebook with details*
