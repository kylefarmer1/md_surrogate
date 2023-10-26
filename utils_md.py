import numpy as np
import tensorflow as tf
import sonnet as snt
from graph_nets import graphs, utils_tf

# TODO: think about moving all potential related functions to another utils file, it's getting rather cluttered
# TODO: Homogenize function variables and descriptions
# TODO: clean up old/commented code
# When working with pairwise potential, the nodes and edges passed around are 2 dimensional (N x M) but
# for ternary they are 3 dimensional (N x _ x M) and the second neighbors are ragged tensors because of the fact that
# not every graph has the same number of second neighbors. Current implementation handles the two cases, but it is
# not the cleanest looking code.
def base_graph(positions, velocities, masses, bonds, parameterization):
    """Define a basic MD system graph structure

    There are (num_atoms) atoms with mass (masses) bonded together, parameterized (parameterization) by a potential
    which is represented as an edge. The graph is directional.
    It is suggested to have the atom with the lower index be the sender and the other to be the receiver.
    For example, in a 3 atom, fully connected system, atom 0 sends an edge to atom 1, 2; atom 1 sends an edge to atom 2.
    The 'links' key in (bonds) will thus have the following value: [[1,2], [2], []].
    The 'edges' key in the returned dictionary is constructed by a look-up table (parameterization dictionary).
    For example if a bond was between atoms of type 1 and type 4. Then that row in the edges key of the returned
    dictionary corresponds to the value of the ['1']['4'] index in parameterization.

    Args:
        positions: np.array(shape = [num_atoms x 2]) with x,y coordinates for each atom
        velocities: np.array(shape = [num_atoms x 2]) with x,y coordinates for each atom
        masses: np.array(shape = [num_atoms x 1]), mass for each atom
        bonds: dictionary with form
        {
            'links': list of lists of indices each atom is bonded to, len = num_atoms
            'atomtypes': list like of integers (representing the atom ID) with size num_atoms for a system.
        }
        parameterization: dictionary of dictionaries of form
        {
            'atomtype_i':
                {
                'atomtype_j': array([1 x num_potential_features]) # i.e. for Morse 1 x 3
                }
        }
        for all atomtypes i where j >= i. In other words, duplicate are not necessary (if [ij] is set, skip [ji]).

    Returns:
        data_dict: dictionary with globals, nodes, edges, receivers, and senders to represent
        a structure like the one above
    """
    # nodes
    # set initial positions, velocities, mass
    n = positions.shape[0]
    nodes = np.zeros((n, 5), dtype=np.float64)
    nodes[:, :2] = positions
    nodes[:, 2:4] = velocities
    nodes[:, 4:5] = masses

    # edges, set interaction parameters from a lookup table
    edges, senders, receivers = [], [], []
    for i, link in enumerate(bonds["links"]):
        for ij, j in enumerate(link):
            atomtype_i = bonds['atomtypes'][i]
            atomtype_j = bonds['atomtypes'][j]
            min, max = np.min([atomtype_i, atomtype_j]), np.max([atomtype_i, atomtype_j])
            edges.append(parameterization[str(min)][str(max)].astype(np.float64))
            senders.append(i)
            receivers.append(bonds["links"][i][ij])

    # globals
    # no globals for now

    return {
        'globals': np.array([0], dtype=np.float64),
        'nodes': nodes,
        'edges': edges,
        'receivers': receivers,
        'senders': senders
    }


def get_example_graph(parameterization):
    """Create a sample system

    :return: a GraphsTuple of a single bounded system. 2 particles 'spinning'
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
        'links': [[1], [0]],
        'atomtypes': np.array([0, 0])
    }
    static_graph = base_graph(positions, velocities, masses, bonds, parameterization)
    graph = utils_tf.data_dicts_to_graphs_tuple([static_graph])
    return static_graph


def get_example_graph_ternary(parameterization):
    """Create a sample system. *** This still needs to be tested ***

    :return: a GraphsTuple of a single bounded system. 3 particles 'spinning'
    """
    positions = np.array([
        [np.sqrt(2), 0.0],
        [0.0, np.sqrt(6)],
        [-np.sqrt(2), 0.0]
    ])
    velocities = np.array([
        [-1.0, -1.0],
        [1.0, 0.0],
        [-1.0, 1.0]
    ])
    v_offset = np.array([
        [0.2, 0.0],
        [0.2, 0.0],
        [0.2, 0.0]
    ])
    theta = 0
    v_rotation = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    velocities = 0.3 * (velocities @ v_rotation.T) + v_offset
    masses = np.array([[1.], [1.], [1.]])
    bonds = {
        'links': [[1, 2], [0, 2], [0, 1]],
        'atomtypes': np.array([0, 0, 0])
    }
    static_graph = base_graph(positions, velocities, masses, bonds, parameterization)
    graph = utils_tf.data_dicts_to_graphs_tuple([static_graph])
    return graph


def get_example_graph_quartet(parameterization):
    """Create a sample system

    :return: a GraphsTuple of a single bounded system. 4 particles 'spinning'
    """
    positions = np.array([
        [1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, -1.0],
        [1.0, -1.0]
    ])
    velocities = np.array([
        [1.0, -1.0],
        [1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, -1.0]
    ])
    v_offset = np.array([
        [0.2, 0.0],
        [0.2, 0.0],
        [0.2, 0.0],
        [0.2, 0.0]
    ])
    theta = -np.pi/2
    v_rotation = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    # ! Why this one do not have rotation
    velocities = 0.5 * velocities + v_offset
    masses = np.array([[1.], [1.], [1.], [1.]])
    bonds = {
        'links': [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]],
        'atomtypes': np.array([0, 0, 0, 0])
    }
    static_graph = base_graph(positions, velocities, masses, bonds, parameterization)
    graph = utils_tf.data_dicts_to_graphs_tuple([static_graph])
    return static_graph


def pairwise_morselike(bond):
    """
    Function to return both the attractive and repulsive components of a Tersoff-like potential function.
    Each component (fa, fr) are expressed in the following:
                f = C_{a,r} * exp[alpha_{a,r} * (r-rc)]
    Thus there are 5 parameters to describe each interaction (excluding the bond order term):
                C_a, C_r, alpha_a, alpha_r, rc
    :param bond: array like [N x M] where N is number of bonds and M is number of features in the parameterization.
                    The indices along the 1st axis should be ordered as [alpha_r, alpha_a, C_r, C_a, rc, r]
    :return: fr, fa - tensor of arrays ([N x 1], [N x 1])
    """
    alpha_r = bond[..., 0:1]
    alpha_a = bond[..., 1:2]
    C_r = bond[..., 2:3]
    C_a = bond[..., 3:4]
    rc = bond[..., 4:5]  # dimer bond distance
    r = bond[..., -1:]  # dimer pairwise distance

    fr = C_r * tf.math.exp(-2 * alpha_r * (r - rc))
    fa = -1 * C_a * 2 * tf.math.exp(-1 * alpha_a * (r - rc))

    return fr, fa


def bond_order_parameter(r1bonds, r2bonds, rc):
    """
    Function to return the bond order component of a Tersoff-like potential function.

    The order of elements along the 1st axis should follow that detailed in the comments of the

    :param r1bonds: array like [N x 1 x M] representing the parameterization of each atomic interaction in the graph between
            the first neighbors (atom i and atom j)
    :param r2bonds: RaggedTensor [N x None x M] representing the parameterization of each atomic interaction in the graph between
            the second neighbors (atom i and atom k)
    :param rc: float; cutoff distance
    :return: b - array like [N x 1]
    """
    c = r2bonds[..., 5:6]  # strength of angular effect
    d = r2bonds[..., 6:7]  # sharpness of angular dependence
    gamma = r2bonds[..., 7:8]  # fitting parameter
    mu = r2bonds[..., 8:9]  # 1/angstroms, could be 0
    h = r2bonds[..., 9:10]  # equilibrium ternary angle

    r1 = r1bonds[..., -1:]  # pairwise distance b/w i & j
    r2 = r2bonds[..., -1:]  # pairwise distance b/w i & k
    theta = r2bonds[..., -2:-1]  # ternary angle

    g = gamma * (1 + c**2/d**2 - c**2/(d**2 + (h + tf.math.cos(theta))**2))
    chi_i = g * tf.math.exp(2 * mu * (r1 - r2))
    # implement ternary cutoff here
    #mask = tf.cast(r2 < rc, dtype=tf.float64)
    mask = cutoff_func(r2, rc)
    chi_i = chi_i * mask
    chi = tf.reduce_sum(chi_i, axis=1, keepdims=True)
    b = (1 + chi) ** (-1 / 2)
    return b


def tersoff_potential(r1bonds, r2bonds, rc):
    """
    Return the energy of interacting atoms following a Tersoff-like potential

    :param r1bonds: arraylike [N x 1 x M] representing the parameterization of each atomic interaction in the graph between
            the first neighbors (atom i and atom j)
    :param r2bonds: RaggedTensor [N x None x M] representing the parameterization of each atomic interaction in the graph between
            the second neighbors (atom i and atom k)
    :param rc: float; cutoff distance
    :return: arraylike [N x 1] representing the pairwise energy of interaction
    """
    fr, fa = pairwise_morselike(r1bonds)
    ternary = bond_order_parameter(r1bonds, r2bonds, rc)
    energy = fr + ternary * fa
    return energy


def morse_potential(bond):
    """Morse function for calculating potential energy

    Calculates the potential energy of an interaction(s) characterized by the morse potential. The morse potential is
    parameterized by u, alpha, r0 and is a function of distance (r); V = V(r, u, alpha, r0).

    :param bond: Tensor of shape E x 4 [u, r0, alpha, r]
    :return: energy per bond: Tensor of shape E x 1
    """
    return bond[..., 0:1] * (tf.math.exp(-2 * bond[..., 2:3] * (bond[..., 3:4] - bond[..., 1:2])) -
                             2 * tf.math.exp(-bond[..., 2:3] * (bond[..., 3:4] - bond[..., 1:2])))


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

def remap_pbc(xlo, xhi, pos):
    """Remaps nodes into system frame due to Periodic Boundary Conditions

    :param xlo: Tensor of shape 2 [x_lo, y_lo]
    :param xhi: Tensor of shape 2 [x_hi, y_hi]
    :param pos: Tensor of shape N x 2 [pos_x, pos_y]
    :return: Tensor of shape N x 2 with updated pos
    """
    lx = xhi - xlo
    mask_hi = tf.cast(pos > xhi, dtype=tf.float64)
    mask_lo = tf.cast(pos < xlo, dtype=tf.float64)

    new_pos = pos + lx * mask_lo
    new_pos = new_pos - lx * mask_hi
    return new_pos


# TODO reformat - always used in the context of node aggregation, no special inputs in _aggregator function
def aggregate(force_per_edge, senders, num_nodes):
    """Sum all the forces acting on each node

    :param force_per_edge: Tensor of shape E x 2 [f_x, f_y]
    :param senders: Tensor of shape E, e.g. graph.senders
    :param num_nodes: Tensor of shape N where N should be the sum of unique values in receivers
    :return: Tensor of shape N x 2 [f_x, f_y]
    """
    _aggregator = tf.math.unsorted_segment_sum
    spring_force_per_node = _aggregator(force_per_edge, senders, num_nodes)

    # newtons third law
    #spring_force_per_node -= _aggregator(force_per_edge, receivers, num_nodes)

    return spring_force_per_node


# TODO revisit - purpose of adding noise? Improve training. Where/how is noise best applied? See comments in func
def apply_noise(graph, edge_noise_level):
    """Applies uniformly-distributed noise to the edges (bonds) of an MD system

    Noise is only applied to the potential parameters (i.e. energy, length, interaction range for Morse potential) of
    the edges.

    :param graph: a GraphsTuple having for some integers N, E, G, Z:
            - nodes: Nx_ Tensor of [_, _, ..., _] for each node
            - edges: ExZ Tensor of [Z_1, Z_2, ..., Z_z] for each edges
            - globals: Gx_ Tensor of [_, _, ..., _] for Global (1 set of attributes per system)
    :param edge_noise_level: Maximum amount to perturb edge spring constants
    :return: The input graph but with noise applied
    """

    # implementation can probably be reduced to just calling tf.random.uniform once on the set of all edges in the graph
    noises = list()
    for edge in graph.n_edge:
        edge_noise = tf.random.uniform(
            [edge, graph.edges.shape[1]],
            minval=-edge_noise_level,
            maxval=edge_noise_level,
            dtype=tf.float64
        )

        #edge_noise = tf.repeat(edge_noise, repeats=edge, axis=0)
        noises.append(edge_noise)

    return graph.replace(edges=graph.edges + tf.concat(noises, axis=0))

# TODO possibly modify the name or description and function variable names
def cutoff_func(r, parameterization):
    """Cutoff function f_c

    A smooth function from 0 to 1 representing the cutoff function in a potential. This particular implementation is
    used in LAMMPS tersoff potential.

    :param r: arraylike [N x 1]
    :param parameterization: float - cutoff distance
    :return: arraylike [N x 1]
    """
    R = parameterization
    D = R * 0.1
    f = 1/2 - 1/2*np.sin(np.pi/2*((r-R)/D))
    U = tf.cast(r >= R-D, tf.bool)
    L = tf.cast(r < R+D, tf.bool)
    B = tf.logical_and(U, L)
    f = f * tf.cast(B, tf.float64) + tf.cast(r < R-D, tf.float64)
    return f


# TODO Possibly reformat the following 5 functions. there are 2 sets of 'ternary'/'pairwise' functions
# that can potentially be consolidated.
def compute_energy_ternary(r1, r2, sender_positions, r1edge, r2edge, rc, lx, func, ke=False):
    """Compute the pairwise energies with ternary interactions on a graph.

    :param r1: arraylike [N x 1 x 2] where N is the number of first-neighbor edges (bonds) in a graph
    :param r2: RaggedTensor [N x None x 2] where N is the number of second-neighbor edges (bonds) in a graph
    :param sender_positions: arraylike [N x 2] where N is the number of edges (bonds) in a graph
    :param r1edge: arraylike [N x 1 x M] where N is the number of first-neighbor edges (interactions) in a graph and M is
            the number of features
    :param r2edge: RaggedTensor [N x None x M] where N is the number of second-neighbor edges (interactions) in a graph and M is
            the number of features
    :param rc: float - cutoff distance
    :param lx: float - system box length
    :param func: function with signature func(bonds) -> arraylike N X 1 where N is the number of edges
    :param ke: bool - calculate and return kinetic energies (True
    :return: pe, ke float - total system energy for all systems in GraphsTruple
    """
    diff = sender_positions - r1
    diffs = list()
    for i in range(2):
        p_x = diff[..., i] - lx * tf.math.rint(diff[..., i] / lx)
        diffs.append(p_x)
    diff1 = tf.stack(diffs, axis=-1)
    x1 = tf.norm(diff1, axis=-1, keepdims=True)

    diff = sender_positions - r2
    diffs = list()
    for i in range(2):
        p_x = diff[..., i] - lx * tf.math.rint(diff[..., i] / lx)
        diffs.append(p_x)
    diff2 = tf.stack(diffs, axis=-1)
    #x2 = tf.norm(diff2, axis=-1, keepdims=True)
    x2 = tf.expand_dims(tf.math.pow(tf.math.add(tf.math.pow(diff[..., 0], 2), tf.math.pow(diff[..., 1], 2)), 0.5), 2)

    angles = compute_angle(diff1, diff2)

    r1bond = tf.concat([r1edge, tf.cast(x1, dtype=tf.float64)], axis=-1)
    r2bond = tf.concat([r2edge, tf.cast(angles, dtype=tf.float64), tf.cast(x2, dtype=tf.float64)], axis=-1)
    energy = func(r1bond, r2bond, rc)
    mask = cutoff_func(x1, rc)
    #mask = tf.cast(x1 < rc, dtype=tf.float64)
    energy = energy * mask
    potential_energy = tf.math.reduce_sum(energy)
    if not ke:
        return energy
    else:
        all_nodes = tf.concat([r1, sender_positions], axis=0)
        unique_nodes = tf.raw_ops.UniqueV2(x=all_nodes, axis=[0])[0]
        kinetic_energy = 0.5 * tf.math.reduce_sum(unique_nodes[..., -1:] * unique_nodes[..., 2:4] ** 2)
        return potential_energy/2, kinetic_energy


# TODO homogenize function input names with ternary func (above)
def compute_energy_pairwise(receiver_positions, sender_positions, edge, rc, lx, func, ke=False):
    """Compute the pairwise energies with ternary interactions on a graph.

    :param receiver_positions: arraylike [N x 2] where N is the number of first-neighbor edges (bonds) in a graph
    :param sender_positions: arraylike [N x 2] where N is the number of edges (bonds) in a graph
    :param edge: arraylike [N x M] where N is the number of first-neighbor edges (interactions) in a graph and M is
            the number of features
    :param rc: float - cutoff distance
    :param lx: float - system box length
    :param func: function with signature func(bonds) -> arraylike N X 1 where N is the number of edges
    :param ke: bool - calculate and return kinetic energies (True
    :return: pe, ke float - total system energy for all systems in GraphsTruple
    """
    diff = sender_positions - receiver_positions
    diffs = list()
    for i in range(2):
        p_x = diff[..., i] - lx * tf.math.rint(diff[..., i] / lx)
        diffs.append(p_x)
    diff = tf.stack(diffs, axis=-1)
    x = tf.norm(diff, axis=-1, keepdims=True)

    bond = tf.concat([edge, x], axis=-1)
    energy = func(bond)
    mask = cutoff_func(x, rc)

    # mask = tf.cast(x < rc, dtype=tf.float64)  # hard cutoff
    energy = energy * mask
    potential_energy = tf.math.reduce_sum(energy)

    if not ke:
        return energy
    else:
        all_nodes = tf.concat([receiver_positions, sender_positions], axis=0)
        unique_nodes = tf.raw_ops.UniqueV2(x=all_nodes, axis=[0])[0]
        kinetic_energy = 0.5 * tf.math.reduce_sum(unique_nodes[..., -1:] * unique_nodes[..., 2:4] ** 2)
        return potential_energy/2, kinetic_energy


# TODO possibly remove, used for troubleshooting/plotting convenience
def compute_energy_pairwise_ind(receiver_positions, sender_positions, edge, rc, lx, func, ke=False):
    """Compute the pairwise energies with ternary interactions on a graph. *Returns pairwise energies for each edge
    (bond) in the system, rather than the total system energy, as seen above.

    :param r1: arraylike [N] where N is the number of first-neighbor edges (interactions) in a graph
    :param r2: arraylike [N] where N is the number of second-neighbor edges (interactions) in a graph
    :param sender_positions: arraylike [N] where N is the number of nodes (atoms) in a graph
    :param r1edge: arraylike [N x M] where N is the number of first-neighbor edges (interactions) in a graph and M is
            the number of features
    :param r2edge: arraylike [N x M] where N is the number of second-neighbor edges (interactions) in a graph and M is
            the number of features
    :param rc: float - cutoff distance
    :param lx: float - system box length
    :param func: function with signature func(bonds) -> arraylike N X 1 where N is the number of edges
    :param ke: bool - calculate and return kinetic energies (True
    :return: pe, ke float - pairwise energies for each edge in the GraphsTuple
    """
    diff = sender_positions - receiver_positions
    diffs = list()
    for i in range(2):
        p_x = diff[..., i] - lx * tf.math.rint(diff[..., i] / lx)
        diffs.append(p_x)
    diff = tf.stack(diffs, axis=-1)
    x = tf.norm(diff, axis=-1, keepdims=True)

    bond = tf.concat([edge, x], axis=-1)
    energy = func(bond)
    mask = cutoff_func(x, rc)
    #mask = tf.cast(x < rc, dtype=tf.float64)
    energy = energy * mask

    return energy


# TODO preprocess_pairwise does not really do much, it is just a little 3 line convenience, but how often is it called?
# The function is called in integration and acceleration
def preprocess_pairwise(graph, next_position):
    """Process the data to compute energies

    Returns the positions of each node, organized by senders and receivers, and returns the edges (just calls
    graph.edges)

    :param graph: GraphsTuple of the current state of the system, only the edges are used for computation
    :param next_position: Tensor of positions with shape [num_atoms x 2]
    :return: Tensor of accelerations with shape [num_atoms x 2]
    """
    receiver_nodes = tf.gather(next_position, graph.receivers)
    sender_nodes = tf.gather(next_position, graph.senders)
    edges = graph.edges

    return sender_nodes, receiver_nodes, edges


# TODO investigate the second nearest neighbor gathering, is there a more direct way of doing it?
def preprocess_ternary(graph, next_position):
    """Process the data to compute energies

    Returns the positions of each node, organized by senders and first and second receivers. Also returns the first
    and second neighbor edges. Compared to the pairwise preprocess function, there is a fair bit of work in gathering
    the second-nearest neighbors

    :param graph: GraphsTuple of the current state of the system, only the edges are used for computation
    :param next_position: Tensor of positions with shape [num_atoms x 2]
    :return: sender_nodes, receiver_nodes, receiver_2_nodes arraylike [N x 2] where N is number of edges
    :return: r1edges, r2edges arraylike [N x M] where N is number of edges and M is number of features in
            parameterization
    """
    # get the r2 neighbors
    #receiver_2 = tf.transpose(tf.concat([tf.gather(graph.receivers, tf.where(graph.senders == r))
    #                                    for r in graph.receivers], axis=1))
    receiver_2 = tf.ragged.stack([tf.squeeze(tf.transpose(tf.gather(graph.receivers, tf.where(graph.senders == r))))
                                  for r in graph.receivers], axis=0)
    s = tf.expand_dims(graph.senders, 1)
    # remove sender as an r2 neighbor
    diff = receiver_2 - s
    mask = tf.cast(diff, dtype=tf.bool)
    r2 = tf.ragged.boolean_mask(diff, mask) + s
    #r2 = tf.reshape(r2, (diff.shape-tf.constant([0, 1]))) + s

    # get edge indices for r2 neighbors
    r2_edge_index = get_ternary_edge_matches(graph.senders, graph.receivers, r2)

    # gather position data
    sender_nodes = tf.gather(next_position, s)
    receiver_nodes = tf.gather(next_position, tf.expand_dims(graph.receivers, axis=1))
    receiver_2_nodes = tf.gather(next_position, r2)

    # gather edge data
    r2edges = tf.gather(graph.edges, r2_edge_index)
    r1edges = tf.expand_dims(graph.edges, 1)

    return sender_nodes, receiver_nodes, receiver_2_nodes, r1edges, r2edges


# TODO again, possibly consolidate ternary/pairwise actions. rename variables2 to something more detailed
def compute_force_autograd(senders, receivers, cutoff, edges, lx, potential, receivers2=None, r2edges=None):
    """ Compute forces from pairwise interactions with automatic differentiation

    :param senders: arraylike [N] where N is the number of edges (bonds) in a graph
    :param receivers: arraylike [N (x 1)] where N is the number of edges (bonds) in a graph, will be 2D [N x 1] in the
            case of ternary interactions
    :param cutoff: float - cutoff distance
    :param edges: arraylike [N x M] where N is the number of first-neighbor edges (interactions) in a graph and M is
            the number of features
    :param lx: float - system box length
    :param potential: function with signature func(bonds) -> arraylike N X 1 where N is the number of edges
    :param receivers2: RaggedTensor [N x None] where N is the number of second-neighbor edges (bonds) in a graph
    :param r2edges: RaggedTensor [N x None x M] where N is the number of second-neighbor edges (interactions) in a graph
            and M is the number of features
    :return: spring_force_per_edge: arraylike [N x 2] where N is number of edges
    :return: energies: arraylike [N x 1] where N is number of edges
    """
    with tf.GradientTape() as tape:
        variables2 = tf.Variable(tf.identity(senders))
        if r2edges is None:
            energies = compute_energy_pairwise(receivers, variables2, edges, cutoff, lx, potential)
        else:
            energies = compute_energy_ternary(receivers, receivers2, variables2, edges,
                                          r2edges, cutoff, lx, potential)
        total_energy = tf.math.reduce_sum(energies)

    spring_force_per_edge = tape.gradient(total_energy, variables2) * -1

    return spring_force_per_edge, energies


# TODO possibly consolidate ternary/pairwise implementations in both euler and verlet integrator functions
# TODO possibly consolidate the number returned parameters, are they all necessary?
def verlet_integrator(graph, cutoff, lx, step_size, acceleration, potential, ternary=False):
    """Velocity-Verlet integrator

    :param graph: GraphsTuple object for a collection of system graphs
    :param cutoff: float - cutoff distance
    :param lx: float - system box length
    :param step_size: float - step size (dt)
    :param acceleration: arraylike [N x 2] where N is number of atoms
    :param potential: function with signature func(bonds) -> arraylike N X 1 where N is the number of edges
    :param ternary: bool
    :return: next_position arraylike: [N x 2] of atoms next positions (at t + 1)
    :return: updated_velocities: [N x 2] of atoms updated velocities (at t+1)
    :return: spring_force_per_node: arraylike [N x 2] where N is number of nodes
    :return: spring_force_per_edge: arraylike [N x 2] where N is number of edges
    :return: energies: arraylike [N x 1] where N is number of edges
    """
    half_velocity = graph.nodes[:, -3:-1] + 0.5 * acceleration * step_size
    next_position = graph.nodes[:, :2] + half_velocity * step_size

    if ternary:
        sender_nodes, receiver_nodes, receiver_2_nodes, r1edges, r2edges = preprocess_ternary(graph, next_position)
        spring_force_per_edge, energies = compute_force_autograd(
            sender_nodes, receiver_nodes, cutoff, r1edges, lx, potential, receiver_2_nodes, r2edges
        )
        spring_force_per_edge = tf.squeeze(spring_force_per_edge)
    else:
        sender_nodes, receiver_nodes, edges = preprocess_pairwise(graph, next_position)
        spring_force_per_edge, energies = compute_force_autograd(
            sender_nodes, receiver_nodes, cutoff, edges, lx, potential
        )
    spring_force_per_node = aggregate(spring_force_per_edge, graph.senders, next_position.shape[0])
    updated_velocities = half_velocity + 0.5 * step_size * spring_force_per_node

    return next_position, updated_velocities, spring_force_per_node, spring_force_per_edge, energies


# TODO update signature (acceleration not used, although may be necessary for consistency when choosing different
#  integrators)
def euler_integrator(graph, cutoff, lx, step_size, acceleration, potential, ternary=False):
    """Euler Integrator

    :param graph: GraphsTuple object for a collection of system graphs
    :param cutoff: float - cutoff distance
    :param lx: float - system box length
    :param step_size: float - step size (dt)
    :param acceleration: arraylike [N x 2] where N is number of atoms
    :param potential: function with signature func(bonds) -> arraylike N X 1 where N is the number of edges
    :param ternary: bool
    :return: updated_position arraylike: [N x 2] of atoms next positions (at t + 1)
    :return: updated_velocities: [N x 2] of atoms updated velocities (at t+1)
    :return: spring_force_per_node: arraylike [N x 2] where N is number of nodes
    """

    if ternary:
        sender_nodes, receiver_nodes, receiver_2_nodes, r1edges, r2edges = preprocess_ternary(graph, graph.nodes[:, :2])
        spring_force_per_edge = compute_force_autograd(
            sender_nodes, receiver_nodes, cutoff, r1edges, lx, potential, receiver_2_nodes, r2edges
        )
        spring_force_per_edge = tf.squeeze(spring_force_per_edge)
    else:
        sender_nodes, receiver_nodes, edges = preprocess_pairwise(graph, graph.nodes[:, :2])
        spring_force_per_edge = compute_force_autograd(
            sender_nodes, receiver_nodes, cutoff, edges, lx, potential
        )

    spring_force_per_node = aggregate(spring_force_per_edge, graph.senders, graph.nodes.shape[0])
    updated_positions = graph.nodes[:, :2] + graph.nodes[:, 2:4] * step_size + 0.5 * spring_force_per_node*step_size**2
    updated_velocities = graph.nodes[:, 2:4] + spring_force_per_node * step_size

    return updated_positions, updated_velocities, spring_force_per_node


# TODO currently only works for a 4 atom system (2 ternary connections), generalize to n ternary connections
def get_ternary_edge_matches(senders, r1, r2):
    """Get the indices of second neighbor edges

    :param senders: arraylike [N] where N is the number of edges (e.g. graph.senders)
    :param r1: arraylike [N] where N is the number of edges (e.g. graph.receivers)
    :param r2: arraylike [N x None] where N is the number of edges & None varies depending on the number of r2
        neighbors for each sender
    :return: RaggedTensor [N x None] where N is the number of edges & None varies depending on the number of r2
        neighbors for each sender
    """
    s = senders.numpy()
    r1 = r1.numpy()
    r2 = r2.numpy()

    srn = np.hstack([s[:, None], r1[:, None]])
    sr2_id = list()
    for row in range(s.size):
        sr2_id_row = np.zeros(shape=(r2[row].size), dtype=np.int32)
        for t in range(r2[row].size):
            sr2_t = np.hstack([s[row, None], r2[row][t:t + 1]])
            match = np.where((srn == sr2_t).all(axis=1))[0][0]
            sr2_id_row[t] = match
        sr2_id.append(sr2_id_row)

    sr2_id = tf.ragged.stack(sr2_id)

    return sr2_id


def compute_angle(d1, d2):
    """Compute the angle between two vectors

    :param d1: tensor [N x 1 x 2]
    :param d2: RaggedTensor [N x None x 2]
    :return: RaggedTensor [N x None x 1]
    """
    dot = tf.math.reduce_sum(tf.math.multiply(d1, d2), axis=2, keepdims=True)
    mag1 = tf.norm(d1, axis=2, keepdims=True)
    mag2 = tf.expand_dims(tf.math.pow(tf.math.add(tf.math.pow(d2[..., 0], 2), tf.math.pow(d2[..., 1], 2)), 0.5), 2)
    return tf.math.acos(dot/(mag1*mag2))


# TODO: update docstring
class MolecularDynamicsSimulatorGraph(snt.Module):  # noqa
    """Implements a basic MD simulator"""

    def __init__(self, step_size, system_size, cutoff, integrator, potential, acceleration_init=None,
                 name="SpringMassSimulator"):
        """Inits MolecularDynamicsSimulatorGraph

        :param step_size: float - step size (dt)
        :param system_size: arraylike [2 x 2] of form [[xlo, ylo],[xhi, yhi]]
        :param cutoff: float - cutoff distance
        :param integrator: function with signature func(graph, cutoff, lx, step_size, acceleration, potential, ternary)
                    -> (arraylike [N x 2], arraylike [N x 2], arraylike [N x 2]) where N is number of nodes (atoms)
        :param potential: function with signature func(bonds) -> arraylike N X 1 where N is the number of edges
        :param acceleration_init: arraylike [N x 2] where N is number of nodes (atoms)
        :param name: string
        """

        super(MolecularDynamicsSimulatorGraph, self).__init__(name=name)
        self._step_size = step_size
        self._cutoff = cutoff
        self._integrator_name = integrator
        self._XLO = system_size[0, :]
        self._XHI = system_size[1, :]
        self._LX = system_size[1, :] - system_size[0, :]
        self._acceleration = acceleration_init
        self._potential = potential
        self._three_body = False
        self._force_per_edge = None
        self._energies = None
        if self._potential.lower() == 'tersoff':
            self._three_body = True

    def __call__(self, graph):
        """

        :param graph: GraphsTuple object for a collection of system graphs
        :return: arraylike [N x 4] positions and velocities of each node (atom) at step t+1
        """
        if self._integrator_name == 'verlet':
            integrator = verlet_integrator
        elif self._integrator_name == 'euler':
            integrator = euler_integrator
        else:
            integrator = None

        if self._three_body:
            ternary = True
            potential = tersoff_potential
        else:
            ternary = False
            potential = morse_potential

        pos, vel, acc, e, e_per_e = integrator(
            graph, self._cutoff, self._LX[0], self._step_size, self._acceleration, potential, ternary=ternary
        )

        # save acc for next call
        self._acceleration = acc
        self._force_per_edge = e
        self._energies = e_per_e

        # remap pbc
        updated_positions = remap_pbc(self._XLO, self._XHI, pos)

        return tf.concat([updated_positions, vel], axis=-1)

    # TODO clean-up function. Some unused variables and misleading variable names
    def get_step_accelerations(self, graph):
        """Get the accelerations for each atom at a specific position a_t = a_t(x_t). This is necessary for the Verlet
        integrator

        :param graph: GraphsTuple object for a collection of system graphs
        :return: arraylike [N x 2] accelerations per node (atom)
        """
        if self._integrator_name == 'verlet':
            integrator = verlet_integrator
        elif self._integrator_name == 'euler':
            integrator = euler_integrator
        else:
            integrator = None

        if self._three_body:
            ternary = True
            potential = tersoff_potential
            sender_nodes, receiver_nodes, receiver_2_nodes, r1edges, r2edges = preprocess_ternary(graph,
                                                                                                  graph.nodes[:, :2])
            spring_force_per_edge, energies = compute_force_autograd(
                sender_nodes, receiver_nodes, self._cutoff, r1edges, self._LX[0], potential, receiver_2_nodes, r2edges
            )
            spring_force_per_edge = tf.squeeze(spring_force_per_edge)
        else:
            ternary = False
            potential = morse_potential
            sender_nodes, receiver_nodes, edges = preprocess_pairwise(graph, graph.nodes[:, :2])
            force_per_edge, energies = compute_force_autograd(
                sender_nodes, receiver_nodes, self._cutoff, edges, self._LX[0], potential
            )
        force_per_node = aggregate(force_per_edge, graph.senders,
                                          graph.nodes.shape[0])
        self._force_per_edge = force_per_edge
        self._energies = energies
        return force_per_node


# TODO `seq_length` has never been used, potentially remove
def rollout_dynamics(simulator, graph, steps, seq_length):
    """Apply some number of Molecular Dynamics steps (`steps`) to an interaction network using tf's `while_loop`.

    :param simulator: A MolecularDynamicsSimulatorGraph, or some module or callable with the same signature
    :param graph: a GraphsTuple having, for some integers N, E, G, Z:
                - nodes: N x 5 Tensor of [x_x, x_y, v_x, v_y, mass] for each atom (node)
                - edges: E x Z Tensor of [Z_1, Z_2, ..., Z_Z] for Z features which parameterize the force potential
    :param steps: int - length of trajectory
    :param seq_length: int - number of previous states to include for the input, i.e.
                nodes_t, nodes_t-1, ..., nodes_t-seq_length
    :return: g: GraphsTuple of the input graph but with noise applied
    :return: nodes_per_step: arraylike [steps+1 x N x 5]  of the node features at each step
    :return: accelerations_per_step: arraylike [steps+1 x N x 2]  of the accelerations of each node at each step
    """

    # TODO possibly consolidate the preprocessing done here into another function (or an already existing one)
    def body(t, graph, nodes_per_step, accelerations_per_step):
        """dynamics step to be used with tf.while_loop

        Provided the system at step t, predict the state at step t+1 following the dynamics provided by the simulator.

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


# TODO homogenize variable names with those found above in `rollout_dynamics`
def generate_trajectory(simulator, graph, steps, edge_noise_level, seq_length):
    """Applies noise and then simulates a molecular dynamics simulation for a number of steps for invoking
    `rollout_dynamics`

    :param simulator: A MolecularDynamicsSimulatorGraph, or some module or callable with the same signature
    :param graph: a GraphsTuple having, for some integers N, E, G, Z:
                - nodes: N x 5 Tensor of [x_x, x_y, v_x, v_y, mass] for each atom (node)
                - edges: E x Z Tensor of [Z_1, Z_2, ..., Z_Z] for Z features which parameterize the force potential
    :param steps: int; length of trajectory
    :param edge_noise_level: float; maximum amount to perturb the bond parameters
    :param seq_length: int; number of previous steps to include for the input, i.e.
                nodes_t, nodes_t-1, ..., nodes_t-seq_length
    :return: g: GraphsTuple of the input graph but with noise applied
    :return: n: arraylike [steps+1 x N x 5]  of the node features at each step
    :return: a: arraylike [steps+1 x N x 2]  of the accelerations of each node at each step
    """

    # implement  noise
    graph = apply_noise(graph, edge_noise_level)
    simulator._acceleration = simulator.get_step_accelerations(graph)
    _, n, a = rollout_dynamics(simulator, graph, steps, seq_length)
    return graph, n, a


def load_weights(model, weights_path):
    """Load model weights with tf checkpoints

    Usage:
        model = learned_simulator(...) # define model architecture
        utils_md.load_weights(model, 'path/to/model/weights' # load weights
        # model is ready for use
    """
    checkpoint = tf.train.Checkpoint(module=model)
    checkpoint.restore(weights_path)


def save_graph(graph, path):
    """Save a GraphsTuple with numpy

    Usage (e.g. saving training data):
        static_graph, num_atoms = gen_data(...) # load example data
        graph = utils_tf.data_dict_to_graphs_tuple(static_graph # convert to GraphsTuple
        utils_md.save_graph(graph, 'path/to/write/graph.npy') # save GraphsTuple
        utils_md.load_graph('path/to/write/graph.npy') # load GraphsTuple
    """
    np.save(path, np.array(graph, dtype=object), allow_pickle=True)


def load_graph(path):
    """Load a GraphsTuple from numpy object

    Usage (e.g. saving training data):
        static_graph, num_atoms = gen_data(...) # load example data
        graph = utils_tf.data_dict_to_graphs_tuple(static_graph # convert to GraphsTuple
        utils_md.save_graph(graph, 'path/to/write/graph.npy') # save GraphsTuple
        utils_md.load_graph('path/to/write/graph.npy') # load GraphsTuple
    """
    keys = ['nodes', 'edges', 'receivers', 'senders', 'globals', 'n_node', 'n_edge']
    loaded_graph_object = np.load(path, allow_pickle=True)
    return graphs.GraphsTuple(**dict(zip(keys, loaded_graph_object)))
