import tensorflow as tf
import sonnet as snt
import numpy as np
from graph_nets import utils_tf
import learned_simulator_graph
import utils_md
import time
import os
from args import Arguments
tf.config.run_functions_eagerly(False)  # turn on for debugging, will run slower

# region Seeds
SEED = 1
np.random.seed(SEED)
tf.random.set_seed(SEED)
rand = np.random.RandomState(SEED)
# endregion Seeds


def create_loss(target, output):
    """ MSE between target and predicted accelerations """
    loss = tf.reduce_mean(tf.reduce_sum((output - target)**2, axis=-1))
    return loss


def create_fully_connected(num_nodes):
    """Create a fully connected graph based on number of atoms

    :param int; num_nodes: number of atoms in system
    :return: list of lists of receiver atoms (indices) for each sender node
    """
    quants = list()
    for i in range(num_nodes):
        temp_quants = list()
        for j in range(i+1, num_nodes):
            temp_quants.append(j)
        quants.append(temp_quants)

    return quants


def get_target_acceleration(position_sequence, velocity, step_size):
    return (position_sequence[-1, ...] - position_sequence[-2, ...] - velocity * step_size) * 2 / (step_size ** 2)


def preprocess_sequence(inputs, t, seq_length):
    """Process training data to be fed to a simulator

    :param inputs: Tensor of shape T x N x V where T is number states in a trajectory, N is number of atoms, and V is
                number of atom attributes
    :param t: int; timestep
    :param seq_length: int; number of previous states to consider
    :return: Tensor of shape N x V where V is updated to reflect the `seq_length` previous states
    """
    # takes the input trajectory and returns the position_velocity_sequence
    mask_sequence = np.arange(t-seq_length, t, dtype=np.int32)
    mask_sequence[mask_sequence < 0] = 0
    velocity_sequence = tf.transpose(tf.gather(inputs[..., 2:4], mask_sequence), [1, 0, 2])
    velocity_sequence = tf.reshape(velocity_sequence, (velocity_sequence.shape[0], -1))
    latest_position = inputs[-1, :, :2]
    return tf.concat([latest_position, velocity_sequence, inputs[-1, :, 4:5]], axis=-1)


def gen_data(batch_size, min_num_atoms, max_num_atoms, system_size):
    """Generate batch of graphs for MD simulations

    Create a batch of initial states as graphs. The number of atoms is sampled from min (inclusive) to max (
    exclusive). The positions are uniformly sampled from the parallelogram that is defined by the system_size. The
    positions are then annealed such that they are not too close to one another. The velocities are uniformly sampled
    from -1.0 to 1.0 for each component. For now, it is assumed all atom types are identical and thus will share
    the same bond parameterization (morse bond with identical parameters).

    :param batch_size: int; number of  systems
    :param min_num_atoms: int; minimum number of atoms to sample from
    :param max_num_atoms: int; maximum number of atoms to sample from
    :param system_size: np.array([[xlo, ylo],[xhi, yhi]) bounds of system box
    :return: static_graph: list of data_dicts in format that can be converted to GraphsTuple
    :return num_atoms: list of number of atoms in each data_dict in static_graph
    """
    morse_energy = 1.0
    morse_length = 1.0
    morse_range = 1.0
    num_atoms = rand.randint(min_num_atoms, max_num_atoms, size=batch_size, dtype=np.int32)

    initial_positions = [rand.uniform(low=system_size[0, :],
                                      high=system_size[1, :],
                                      size=(num, 2))
                         for num in num_atoms]
    velocities = [rand.uniform(low=-1.0, high=1.0, size=(num, 2))
                  for num in num_atoms]

    masses = [np.ones((num, 1)) for num in num_atoms]

    bonds = list()
    for num in num_atoms:
        links = create_fully_connected(num)
        energies = [[morse_energy for _ in sublist]for sublist in links]
        lengths = [[morse_length for _ in sublist]for sublist in links]
        interaction_range = [[morse_range for _ in sublist] for sublist in links]
        bond = {
            'links': links,
            'energies': energies,
            'lengths': lengths,
            'interaction_ranges': interaction_range
        }
        bonds.append(bond)

    # anneal positions
    def anneal(positions):
        """Anneal atoms such that they are not too close to one another

        Rather than actually annealing (moving particles away from each other), particles are simply resampled
        recursively until they are no longer within a set distance from any other particle. The distance is currently
        hard coded to be 0.8 * morse bond length.

        :param positions: np.array(shape=[num_atoms x 2]) of atom positions
        :return: np.array(shape=[num_atoms x 2]) of atom positions
        """
        for i in range(positions.shape[0]):
            for j in range(i+1, positions.shape[0]):
                dist = np.linalg.norm(positions[j]-positions[i])
                if dist < 0.8 * morse_length:
                    positions[j] = rand.uniform(low=system_size[0, :],
                                                high=system_size[1, :],
                                                size=(1, 2))
                    anneal(positions)
        return positions

    positions = [anneal(position) for position in initial_positions]

    static_graphs = [
        utils_md.base_graph(pos, vel, mass, bond)
        for pos, vel, mass, bond in zip(positions, velocities, masses, bonds)]

    return static_graphs, num_atoms


def train(args):
    # region Model definition
    num_time_steps = args.num_time_steps
    system_size = np.array([
        [args.unit_cell.xlo, args.unit_cell.ylo],
        [args.unit_cell.xhi, args.unit_cell.yhi]
    ])
    step_size = args.step_size
    seq_length = 1
    num_dimensions = 2
    model_kwargs = dict(
        latent_size=args.latent_size,
        mlp_hidden_size=args.mlp_hidden_size,
        mlp_num_hidden_layers=args.mlp_num_hidden_layers,
        num_message_passing_steps=args.num_message_passing_steps)
    connectivity_radius = args.cutoff
    model = learned_simulator_graph.LearnedSimulator(
        num_dimensions, step_size, system_size, connectivity_radius, model_kwargs)
    simulator = utils_md.MolecularDynamicsSimulatorGraph(
        step_size=step_size, system_size=system_size, cutoff=connectivity_radius, integrator='verlet')

    # load_path = 'models/md_2_5.00e+05_50_0.1_7/weights-1'
    # utils_md.load_weights(model, load_path)
    # endregion Model definition

    # region Data
    # region train set
    static_graph_tr, num_atoms_tr = gen_data(
        args.training_set.batch_size, args.training_set.min_num_atoms, args.training_set.max_num_atoms + 1, system_size)
    base_graph_tr = utils_tf.data_dicts_to_graphs_tuple(static_graph_tr)
    simulator._acceleration = simulator.get_step_accelerations(base_graph_tr, base_graph_tr.nodes[..., :2])
    initial_conditions_tr, true_trajectory_tr, true_accelerations_tr = utils_md.generate_trajectory(
        simulator,
        base_graph_tr,
        num_time_steps,
        edge_noise_level=0.1,
        seq_length=seq_length
    )
    # save training_data
    training_data_save_path = f'./data/training_set_{args.training_set.batch_size}_' \
                              f'{args.training_set.min_num_atoms}_' \
                              f'{args.training_set.max_num_atoms}_{system_size[0,0]}_{system_size[1,1]}.npy'
    utils_md.save_graph(initial_conditions_tr, training_data_save_path)
    print('saved training dataset to ', training_data_save_path)
    # endregion train set

    # region test set
    static_graph_ge, num_atoms_ge = gen_data(args.training_set.batch_size, args.training_set.min_num_atoms,
                                             args.training_set.max_num_atoms + 1, system_size)
    base_graph_ge = utils_tf.data_dicts_to_graphs_tuple(static_graph_ge)
    simulator._acceleration = simulator.get_step_accelerations(base_graph_ge, base_graph_ge.nodes[..., :2])
    initial_conditions_ge, true_trajectory_ge, true_accelerations_ge = utils_md.generate_trajectory(
        simulator,
        base_graph_ge,
        num_time_steps,
        edge_noise_level=0.1,
        seq_length=seq_length
    )
    # endregion test set
    # endregion Data

    # region optimizer
    optimizer = snt.optimizers.Adam(learning_rate=args.learning_rate)

    def update_step(graph, next_positions, target_accelerations):

        with tf.GradientTape() as tape:
            pred_accelerations = model.get_step_accelerations(graph, next_positions)
            loss_tr = create_loss(target_accelerations, pred_accelerations)

        gradients = tape.gradient(loss_tr, model.trainable_variables)
        optimizer.apply(gradients, model.trainable_variables)
        return pred_accelerations, loss_tr

    sample_input = preprocess_sequence(true_trajectory_tr, 0, seq_length)
    sample_input_graph = initial_conditions_tr.replace(nodes=sample_input)
    input_signature = [
        utils_tf.specs_from_graphs_tuple(sample_input_graph),
        tf.TensorSpec(shape=list(true_trajectory_tr[0, :, :2].shape),
                      dtype=true_trajectory_tr[0, :, :2].dtype),
        tf.TensorSpec(shape=list(true_trajectory_tr[0, :, :2].shape),
                      dtype=true_trajectory_tr[0, :, :2].dtype)
    ]

    compiled_update_step = tf.function(update_step, input_signature=input_signature)
    # endregion optimizer

    # region logs
    last_iteration = 0
    losses_tr = list()
    losses_ge = list()
    # endregion logs

    # region training
    print("# (iteration number), " 
          "T (elapsed seconds), "
          "Lt (training 1-step loss), "
          "Lge (test/generalization rollout loss)")

    start_time = time.time()
    for iteration in range(last_iteration, int(args.num_training_iterations)):
        t = tf.random.uniform([], minval=0, maxval=num_time_steps - 2, dtype=tf.int32)
        inputs_pos_vel = preprocess_sequence(true_trajectory_tr, t + 1, seq_length)
        next_positions = true_trajectory_tr[t + 1, :, :2]
        graph = initial_conditions_tr.replace(nodes=inputs_pos_vel)
        target_acceleration = simulator.get_step_accelerations(graph, next_positions)
        # set current acceleration from simulator
        # model._acceleration = simulator.get_step_accelerations(graph, true_trajectory_tr[t, :, :2])
        outputs_tr, loss_tr = compiled_update_step(graph, next_positions, target_acceleration)
        if iteration % args.log_increment == 0:
            model._acceleration = simulator.get_step_accelerations(initial_conditions_ge,
                                                                   initial_conditions_ge.nodes[..., :2])
            _, predicted_nodes_rollout_ge, pred_accelerations = utils_md.rollout_dynamics(
                model, initial_conditions_ge,  num_time_steps, seq_length)

            loss_ge = create_loss(true_accelerations_ge, pred_accelerations)

            losses_tr.append(loss_tr.numpy())
            losses_ge.append(loss_ge.numpy())
            elapsed = time.time() - start_time
            print("# {}, {:06d}, t {:.1f}, Lt {:.4f}, Lge {:.4f}".format(
                t, iteration, elapsed, loss_tr.numpy(), loss_ge.numpy(),
            ))
    # endregion training

    # region save model weights
    model_save_directory = args.model_save_directory.parent
    if not os.path.exists(model_save_directory):
        os.makedirs(model_save_directory)
    save_path = model_save_directory + args.model_save_directory.child
    checkpoint = tf.train.Checkpoint(module=model)
    print(checkpoint.save(save_path))
    # endregion save model weights


if __name__ == '__main__':
    arg_parser = Arguments()
    args = arg_parser.parse_args()
    Arguments().print_args(args)
    train(args)
