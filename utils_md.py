import numpy as np
import tensorflow as tf
import sonnet as snt
from graph_nets import graphs, utils_tf


def base_graph(positions, velocities, masses, bonds):
    """Define a basic MD system graph structure

    There are (num_atoms) atoms with mass (masses) bonded together, parameterized by a Morse potential which is
    represented as an edge. The graph is directional and each atom should only have a single edge to each unique
    atom. It is suggested to have the atom with the lower index be the sender and the other to be the receiver. For
    example, in a 3 atom, fully connected system, atom 0 sends an edge to atom 1, 2; atom 1 sends an edge to atom 2.
    The 'links' key in (bonds) will thus have the following value: [[1,2], [2], []].

    Args:
        positions: np.array(shape = [num_atoms x 2]) with x,y coordinates for each atom
        velocities: np.array(shape = [num_atoms x 2]) with x,y coordinates for each atom
        masses: np.array(shape = [num_atoms x 1]), mass for each atom
        bonds: dictionary with form
        {
            'links': list of lists of indices each atom is bonded to, len = num_atoms
            'energies': list of lists of energies for each bond, len = num_atoms
            'lengths': list of lists of morse bond lengths for each bond, len = num_atoms
            'interaction_range': list of lists of morse interactions range, len = num_atoms
        }

    Returns:
        data_dict: dictionary with globals, nodes, edges, receivers, and senders to represent
        a structure like the one above
    """
    # nodes
    # set initial positions, velocities, mass
    n = positions.shape[0]
    nodes = np.zeros((n, 5), dtype=np.float32)
    nodes[:, :2] = positions
    nodes[:, 2:4] = velocities
    nodes[:, 4:5] = masses

    # edges
    # set morse bond, morse energy, morse interaction range
    edges, senders, receivers = [], [], []
    for i, link in enumerate(bonds['links']):
        for j in range(len(link)):
            edges.append([bonds['energies'][i][j], bonds['lengths'][i][j],
                          bonds['interaction_ranges'][i][j]])
            senders.append(i)
            receivers.append(bonds['links'][i][j])

    # globals
    # no globals for now

    return {
        'globals': [],
        'nodes': nodes,
        'edges': edges,
        'receivers': receivers,
        'senders': senders
    }


def get_example_graph():
    """Create a sample system

    :return: a GraphsTuple of a bounded system. 2 particles 'spinning'
    """
    positions = np.array([
        [1.0, 1.0],
        [-1.0, -1.0]
    ])
    velocities = np.array([
        [1.0, -1.0],
        [-1.0, 1.0]
    ])
    v_offset = np.array([
        [0.2, 0.0],
        [0.2, 0.0]
    ])
    velocities = 0.3 * velocities + v_offset
    masses = np.array([[1.], [1.]])
    bonds = {
        'links': [[1], []],
        'energies': [[1.0], []],
        'lengths': [[1.0], []],
        'interaction_ranges': [[1.0], []]
    }
    static_graph = base_graph(positions, velocities, masses, bonds)
    graph = utils_tf.data_dicts_to_graphs_tuple([static_graph])
    return graph


def morse_potential(bond):
    """Morse function for calculating potential energy

    Calculates the potential energy of an interaction(s) characterized by the morse potential. The morse potential is
    parameterized by u, alpha, r0 and is a function of distance (r); V = V(r, u, alpha, r0).

    :param bond: Tensor of shape E x 4 [u, r0, alpha, r]
    :return: energy per bond: Tensor of shape E x 1
    """
    return bond[..., 0:1] * (np.exp(-2 * bond[..., 2:3] * (bond[..., 3:4] - bond[..., 1:2])) -
                             2 * np.exp(-bond[..., 2:3] * (bond[..., 3:4] - bond[..., 1:2])))


def grad_morse_potential(bond):
    """Gradient of the morse function for calculating interaction forces

    Calculates the force of an interaction(s) characterized by the morse potential. The morse potential is
    parameterized by u, alpha, r0 and is a function of distance (r); V = V(r, u, alpha, r0).

    :param bond: Tensor of shape E x 4 [u, r0, alpha, r]
    :return: force per bond: Tensor of shape E x 1
    """
    return -2 * bond[..., 0:1] * bond[..., 2:3] * \
        (tf.math.exp(-2 * bond[..., 2:3] * (bond[..., 3:4] - bond[..., 1:2])) -
         tf.math.exp(-1 * bond[..., 2:3] * (bond[..., 3:4] - bond[..., 1:2])))


def velocity_verlet_integration_pos(nodes, acceleration, step_size):
    """Velocity Verlet Integrator part 1

    Update the positions and half velocities (steps 1 & 2) with the Velocity Verlet integrator:

    Standard implementation of the Velocity-Verlet integrator:
    1. Calculate half velocity: v(t+1/2dt) = v(t) + 1/2a(t)dt
    2. Update position x(t+dt) = x(t) + v(t+1/2dt)dt
    3. Calculate accelerations(forces) (a(t+dt)) with x(t+dt)
    4. Update velocity v(t+dt) = v(t+1/2dt) + 1/2a(t+dt)dt

    :param nodes: Tensor of shape N x 4 [pos_x, pos_y, vel_x, vel_y]
    :param acceleration: Tensor of shape N x 2 [acc_x, acc_y]
    :param step_size: float
    :return: list of 2 Tensors [Tensor(pos_x, pos_y), Tensor(vel_x, vel_y)]
    """
    half_vel = nodes[..., 2:4] + 0.5 * acceleration * step_size
    new_pos = nodes[..., :2] + half_vel * step_size
    return new_pos, half_vel


def velocity_verlet_integration_vel(half_velocity, acceleration, step_size):
    """Velocity Verlet Integrator part 2

    Update the velocities (step 4) with the Velocity Verlet integrator:

    Standard implementation of the Velocity-Verlet integrator:
    1. Calculate half velocity: v(t+1/2dt) = v(t) + 1/2a(t)dt
    2. Update position x(t+dt) = x(t) + v(t+1/2dt)dt
    3. Calculate accelerations(forces) (a(t+dt)) with x(t+dt)
    4. Update velocity v(t+dt) = v(t+1/2dt) + 1/2a(t+dt)dt

    :param half_velocity: Tensor of shape N x 2 [vel_x, vel_y]
    :param acceleration: Tensor of shape N x 2 [acc_x, acc_y]
    :param step_size: float
    :return: Tensor of shape N x 2 [vel_x, vel_y]
    """
    new_vel = half_velocity + 0.5 * step_size * acceleration
    return new_vel


def euler_integration(nodes, acceleration, step_size):
    """Euler Integrator

    :param nodes: Tensor of shape N x 4 [pos_x, pos_y, vel_x, vel_y]
    :param acceleration: Tensor of shape N x 2 [acc_x, acc_y]
    :param step_size: float
    :return: updated positions and velocities: list of 2 Tensors [Tensor(pos_x, pos_y), Tensor(vel_x, vel_y)]
    """
    new_pos = nodes[..., :2] + nodes[..., 2:4] * step_size + 0.5 * acceleration * step_size ** 2
    new_vel = nodes[..., 2:4] + acceleration * step_size
    return new_pos, new_vel


def remap_pbc(xlo, xhi, pos):
    """Remaps nodes into system frame due to Periodic Boundary Conditions

    :param xlo: Tensor of shape 2 [x_lo, y_lo]
    :param xhi: Tensor of shape 2 [x_hi, y_hi]
    :param pos: Tensor of shape N x 2 [pos_x, pos_y]
    :return: Tensor of shape N x 2 with updated pos
    """
    lx = xhi - xlo
    mask_hi = tf.cast(pos > xhi, dtype=tf.float32)
    mask_lo = tf.cast(pos < xlo, dtype=tf.float32)

    new_pos = pos + lx * mask_lo
    new_pos = new_pos - lx * mask_hi
    return new_pos


def aggregate(force_per_edge, senders, receivers, num_nodes):
    """Sum all the forces acting on each node

    :param force_per_edge: Tensor of shape E x 2 [f_x, f_y]
    :param senders: Tensor of shape E, e.g. graph.senders
    :param receivers: Tensor of shape E, e.g. graph.receivers
    :param num_nodes: Tensor of shape N where N should be the sum of unique values in receivers
    :return: Tensor of shape N x 2 [f_x, f_y]
    """
    _aggregator = tf.math.unsorted_segment_sum
    spring_force_per_node = _aggregator(force_per_edge, senders, num_nodes)

    # newtons third law
    spring_force_per_node -= _aggregator(force_per_edge, receivers, num_nodes)

    return spring_force_per_node


def apply_noise(graph, edge_noise_level):
    """Applies uniformly-distributed noise to the edges (bonds) of an MD system

    Noise is only applied to the potential parameters (i.e. energy, length, interaction range for Morse potential) of
    the edges. The noise is applied such that each bond in a unique system has the same noise, but different systems
    have different noises.

    :param graph: a GraphsTuple having for some integers N, E, G, Z:
            - nodes: Nx_ Tensor of [_, _, ..., _] for each node
            - edges: ExZ Tensor of [Z_1, Z_2, ..., Z_z] for each edges
            - globals: Gx_ Tensor of [_, _, ..., _] for Global (1 set of attributes per system)
    :param edge_noise_level: Maximum amount to perturb edge spring constants
    :return: The input graph but with noise applied
    """
    noises = list()
    for edge in graph.n_edge:
        edge_noise = tf.random.uniform(
            [1, graph.edges.shape[1]],
            minval=-edge_noise_level,
            maxval=edge_noise_level)
        edge_noise = tf.repeat(edge_noise, repeats=edge, axis=0)
        noises.append(edge_noise)

    return graph.replace(edges=graph.edges + tf.concat(noises, axis=0))


def compute_forces(receiver_nodes, sender_nodes, edge, rc, LX, func):
    """Compute the forces per bond (edge) for a system state

    Forces are a function of distance and parameterized by a potential (func). Periodic boundary conditions are
    assumed, thus the closest periodic image of each node is used calculate distance.

    :param receiver_nodes: Tensor of indices of receiver nodes
    :param sender_nodes: Tensor of indices of sender nodes
    :param edge: Tensor of shape [num_edges x num_edge_features] that parameterize func (for Morse, num_edge_features=3)
    :param rc: float; cutoff distance to ignore force contribution
    :param LX: np.array([width, height]) of system
    :param func: gradient of potential used to calculate the forces, signature should accept a Tensor which
            parameterizes each bond
    :return: Tensor of shape [num_edges x 2]; acceleration vector (a_x, a_y) for each edge.
    """
    # find the distance between each node in an edge (implement the closest periodic image of j)
    diff = receiver_nodes[..., 0:2] - sender_nodes[..., 0:2]
    diffs = list()
    for i in range(2):
        p_x = diff[..., i] - LX * tf.math.rint(diff[..., i] / LX)
        diffs.append(p_x)
    diff = tf.stack(diffs, axis=-1)
    x = tf.norm(diff, axis=-1, keepdims=True)
    xhat = diff / x

    # compute force
    bond = tf.concat([edge, x], axis=-1)
    force_magnitude = func(bond)
    force = force_magnitude * xhat
    mask = tf.cast(x < rc, dtype=tf.float32)
    force = force * mask
    return force


def compute_energies(receiver_nodes, sender_nodes, edge, rc, LX, func):

    diff = receiver_nodes[..., 0:2] - sender_nodes[..., 0:2]
    diffs = list()
    for i in range(2):
        p_x = diff[..., i] - LX * tf.math.rint(diff[..., i] / LX)
        diffs.append(p_x)
    diff = tf.stack(diffs, axis=-1)
    x = tf.norm(diff, axis=-1, keepdims=True)

    bond = tf.concat([edge, x], axis=-1)
    energy = func(bond)
    mask = tf.cast(x < rc, dtype=tf.float32)
    energy = energy * mask
    potential_energy = tf.math.reduce_sum(energy)

    all_nodes = tf.concat([receiver_nodes, sender_nodes], axis=0)
    unique_nodes = tf.raw_ops.UniqueV2(x=all_nodes, axis=[0])[0]
    kinetic_energy = 0.5 * tf.math.reduce_sum(unique_nodes[..., -1:] * unique_nodes[..., 2:4]**2)

    return potential_energy, kinetic_energy


class MolecularDynamicsSimulatorGraph(snt.Module):  # noqa
    """Implements a basic MD simulator

    Attributes:
        _step_size: step size for integrator
        _cutoff: distance to start ignoring interactions
        _integrator_name: name of integrator to update system state, either 'verlet' or 'euler'
        _acceleration: current acceleration vector for each atom in the system
        _XLO, _XHI, _LX: system bounds and lengths, determined by `system_size`
    """

    def __init__(self, step_size, system_size, cutoff, integrator, acceleration_init=None, name="SpringMassSimulator"):
        """Inits MolecularDynamicsSimulatorGraph

        Args:
              step_size: float
              system_size: np.array([[xlo, ylo], [xhi, yhi]]
              cutoff: float
              integrator: str - either 'verlet' or 'euler'
              acceleration_init: np.array(shape=[num_atoms x 2])
        """
        super(MolecularDynamicsSimulatorGraph, self).__init__(name=name)
        self._step_size = step_size
        self._cutoff = cutoff
        self._integrator_name = integrator
        self._XLO = system_size[0, :]
        self._XHI = system_size[1, :]
        self._LX = system_size[1, :] - system_size[0, :]
        self._acceleration = acceleration_init

    def __call__(self, graph):
        """Apply one step of molecular dynamics

        # TODO make potential function modular (like the integrator)
        Positions and velocities are updated according to the integrator scheme chosen (euler or verlet) and forces
        are calculated by the chosen potential function

        :param graph: a GraphsTuple having, for some integers N, E, G, Z:
                - nodes: N x Z Tensor of [x_x, x_y, v_x1, v_y1, v_x2, v_x2, mass] for each atom (node). The second to
                    last and third to least columns should represent the velocity (x and y component) of the system at
                    the previous timestep
                - edges: E x Z Tensor of [Z_1, Z_2, ..., Z_Z] for Z features which parameterize the force potential
        :return: A tensor with shape N x 4 of [x_x, x_y, v_x, v_y] that describes the state of the system after 1 time
                step
        """

        # TODO modify to accommodate euler integrator
        half_velocity = graph.nodes[:, -3:-1] + 0.5 * self._acceleration * self._step_size
        next_position = graph.nodes[:, :2] + half_velocity * self._step_size

        integrator = velocity_verlet_integration_vel

        spring_force_per_node = self._preprocess(graph, next_position)
        self._acceleration = spring_force_per_node

        # integrate forces
        updated_velocities = integrator(
            half_velocity, spring_force_per_node, self._step_size)
        updated_positions = remap_pbc(self._XLO, self._XHI, next_position)

        # update graph
        return tf.concat([updated_positions, updated_velocities], axis=-1)

    # TODO reorganize the following two functions as the second one simply just returns the first one with same
    #  signature
    def _preprocess(self, graph, next_position):
        """Process the data to compute forces

        Function name is outdated. Force per bond is calculated and then applied to each node. Senders exert a
        positive force on the receiver and receivers exert a negative force (Newton's third law) on the sender. For
        Verlet integration, a(t + dt) is a function of x(t + dt) (positions must be updated first) For Euler
        integration a(t+dt) is a function of x(t) (positions are updated after computing forces)

        :param graph: GraphsTuple of the current state of the system, only the edges are used for computation
        :param next_position: Tensor of positions with shape [num_atoms x 2]
        :return: Tensor of accelerations with shape [num_atoms x2]
        """
        receiver_nodes = tf.gather(next_position, graph.receivers)
        sender_nodes = tf.gather(next_position, graph.senders)

        # calculate forces from previous step
        spring_force_per_edge = compute_forces(receiver_nodes, sender_nodes,
                                               graph.edges,
                                               self._cutoff,
                                               self._LX[0],
                                               grad_morse_potential)

        # aggregate forces per atom
        spring_force_per_node = aggregate(spring_force_per_edge, graph.senders, graph.receivers, next_position.shape[0])
        return spring_force_per_node

    def get_step_accelerations(self, graph, next_position):
        """Get the accelerations for each atom at a specific position a_t = a_t(x_t)

        For Verlet integration, a(t + dt) is a function of x(t + dt) (positions must be updated first)
        For Euler integration a(t+dt) is a function of x(t) (positions are updated after computing forces)

        :param graph: GraphsTuple of the current state of the system, only the edges are used for computation
        :param next_position: Tensor of positions with shape [num_atoms x 2]
        :return: Tensor of accelerations with shape [num_atoms x2]
        """
        spring_force_per_node = self._preprocess(graph, next_position)
        return spring_force_per_node


def rollout_dynamics(simulator, graph, steps, seq_length):
    """Apply some number of Molecular Dynamics steps to an interaction network



    :param simulator: A MolecularDynamicsSimulatorGraph, or some module or callable with the same signature
    :param graph: a GraphsTuple having, for some integers N, E, G, Z:
                - nodes: N x 5 Tensor of [x_x, x_y, v_x, v_y, mass] for each atom (node)
                - edges: E x Z Tensor of [Z_1, Z_2, ..., Z_Z] for Z features which parameterize the force potential
    :param steps: int; length of trajectory
    :param seq_length: int; number of previous states to include for the input, i.e.
                nodes_t, nodes_t-1, ..., nodes_t-seq_length
    :return: graph: the input graph but with noise applied
    :return: n: a `steps+1`xNx5 Tensor of the node features at each step
    :return: a: a `steps+1xNx2 Tensor of the accelerations of each node at each step
    """

    def body(t, graph, nodes_per_step, accelerations_per_step):
        """dynamics step to be used with tf.while_loop

        Provided the system at step t-1, predict the state at step t following the dynamics provided by the simulator.
        The inputs processing should follow closely to that in train_md._preprocess. That function cannot be used here
        due to the use of the TensorArray object which has different indexing properties.

        :param t: int; timestep
        :param graph: see above
        :param nodes_per_step: TensorArray defined below which will hold the predicted positions and velocities
        :param accelerations_per_step: TensorArray defined below which will hold the predicted accelerations
        :return: t+1: int; the next step
        :return: graph: GraphsTuple; updated system
        :return: nodes_per_step: TensorArray updated at time t
        :return: accelerations_per_step: TensorArray updated at time t
        """
        # get sequence of positions and velocities
        mask_sequence = np.arange(t - seq_length, t, dtype=np.int32)
        mask_sequence[mask_sequence < 0] = 0  # if the sequence goes into negative times, just append the initial state
        position_velocity_sequence = nodes_per_step.gather(mask_sequence)  # apply the mask to the nodes
        position_velocity_sequence = tf.transpose(position_velocity_sequence, [1, 0, 2])  # axes need to match simulator
        velocity_sequence = position_velocity_sequence[:, :, 2:-1]  # get velocity sequence which will be 'flattened'

        # spread the history of velocities for each node as node features
        # they are spread such that the latest velocities are along the right most columns
        velocity_sequence = tf.reshape(velocity_sequence, (velocity_sequence.shape[0], -1))

        # rewrite graph nodes for use in the simulator
        new_nodes = tf.concat([
            graph.nodes[:, :2], velocity_sequence, graph.nodes[:, 4:5]
        ], axis=-1)
        graph = graph.replace(nodes=new_nodes)

        # predicted next position with simulator
        predicted_pos_vel = simulator(graph)
        if isinstance(predicted_pos_vel, list):
            # TODO investigate below, check the model architecture
            # i think this is necessary for when there are many processing steps, but not entirely sure
            predicted_pos_vel = predicted_pos_vel[-1]

        # update graph with next state
        graph = graph.replace(nodes=tf.concat([
            predicted_pos_vel, graph.nodes[..., 4:5]], axis=-1))

        return t + 1, graph, nodes_per_step.write(t, graph.nodes), \
            accelerations_per_step.write(t, simulator._acceleration)

    accelerations_per_step = tf.TensorArray(
        dtype=graph.nodes.dtype, size=steps + 1, element_shape=simulator._acceleration.shape
    )
    nodes_per_step = tf.TensorArray(
        dtype=graph.nodes.dtype, size=steps + 1, element_shape=graph.nodes.shape
    )
    accelerations_per_step = accelerations_per_step.write(0, simulator._acceleration)
    nodes_per_step = nodes_per_step.write(0, graph.nodes)

    _, g, nodes_per_step, accelerations_per_step = tf.while_loop(
        lambda t, *unused_args: t <= steps,
        body,
        loop_vars=[1, graph, nodes_per_step, accelerations_per_step]
    )
    return g, nodes_per_step.stack(), accelerations_per_step.stack()


def generate_trajectory(simulator, graph, steps, edge_noise_level, seq_length):
    """Applies noise and then simulates a molecular dynamics simulation for a number of steps

    :param simulator: A MolecularDynamicsSimulatorGraph, or some module or callable with the same signature
    :param graph: a GraphsTuple having, for some integers N, E, G, Z:
                - nodes: N x 5 Tensor of [x_x, x_y, v_x, v_y, mass] for each atom (node)
                - edges: E x Z Tensor of [Z_1, Z_2, ..., Z_Z] for Z features which parameterize the force potential
    :param steps: int; length of trajectory
    :param edge_noise_level: float; maximum amount to perturb the bond parameters
    :param seq_length: int; number of previous steps to include for the input, i.e.
                nodes_t, nodes_t-1, ..., nodes_t-seq_length
    :return: graph: the input graph but with noise applied
    :return: n: a `steps+1`xNx5 Tensor of the node features at each step
    :return: a: a `steps+1xNx2 Tensor of the accelerations of each node at each step
    """

    # implement  noise
    graph = apply_noise(graph, edge_noise_level)
    simulator._acceleration = simulator.get_step_accelerations(graph, graph.nodes[..., :2])
    _, n, a = rollout_dynamics(simulator, graph, steps, seq_length)
    return graph, n, a


def load_weights(model, weights_path):
    """Load model weights with tf checkpoints"""
    checkpoint = tf.train.Checkpoint(module=model)
    checkpoint.restore(weights_path)


@tf.function(reduce_retracing=True)
def compute_n_edge(senders, n_node):
    """ Compute the number of edges per graph in the batch

    After modifying the connections of the graph (such as when applying a cutoff), the number of edges ber batch must
    be recomputed. With tf2, AutoGraph is able to compile control flow statements such as `if`, but must be wrapped
    in a tf.function(). Reduce_retracing is required because the shape of the arguments is not consistent across calls.

    :param senders: Tensor of shape E, e.g. graph.senders
    :param n_node: Tensor of shape B, where B is the batch size i.e. the number of graphs in the GraphsTuple. e.g.
            graph.n_node. each value in n_node represents the number of nodes in each graph in the batch
    :return: Tensor of shape B, updated number of edges in each graph after applying cutoffs
    """
    if tf.equal(tf.size(senders), 0):
        return tf.zeros(1, dtype=tf.int64)
    else:
        cumsum = tf.math.cumsum(n_node)
        diff1 = senders[:, None] < cumsum[None, :]
        diff1 = tf.concat([diff1, tf.zeros((1, tf.size(cumsum)), dtype=tf.bool)], axis=0)
        diff1 = tf.cast(diff1, tf.int16)
        diff1 = diff1[:-1] - diff1[1:]
        n = tf.experimental.numpy.nonzero(diff1)[0]
        n = n + 1
        n = tf.squeeze(tf.concat([tf.zeros((1, 1), dtype=tf.int64), n[:, None]], axis=0))
        leading_diff = tf.cast(senders[0] < cumsum, dtype=tf.int64)
        split_lo = tf.math.count_nonzero(leading_diff == 0)
        diff2 = n[1:] - n[:-1]
        return tf.concat([tf.zeros(split_lo, dtype=tf.int64), diff2], axis=0)


def save_graph(graph, path):
    """Save a GraphsTuple with numpy"""
    np.save(path, np.array(graph, dtype=object), allow_pickle=True)


def load_graph(path):
    """Load a GraphsTuple from numpy object"""
    keys = ['nodes', 'edges', 'receivers', 'senders', 'globals', 'n_node', 'n_edge']
    loaded_graph_object = np.load(path, allow_pickle=True)
    return graphs.GraphsTuple(**dict(zip(keys, loaded_graph_object)))
