import graph_model_global
import graph_nets as gn
import sonnet as snt
import tensorflow as tf
import utils_md


class LearnedSimulator(snt.Module):  # noqa
    """Learned simulator from https://arxiv.org/pdf/2002.09405.pdf."""

    def __init__(
            self,
            num_dimensions,
            step_size,
            system_size,
            connectivity_radius,
            graph_network_kwargs,
            acceleration_init=None,
            name="LearnedSimulator"):
        """Initializes the model

        :param num_dimensions: Dimensionality of the problem (2D)
        :param step_size: step size to apply to integrator
        :param system_size: np.array([[xlo, ylo],[xhi, yhi]]
        :param connectivity_radius: cutoff distance to ignore interactions
        :param graph_network_kwargs: Keyword arguments to pass to the learned part
            of the graph network `model.EncodeProcessDecode`.
        :param acceleration_init: Tensor N x 2 of accelerations for each atom at t=0. Necessary for Verlet integrator
        :param name: Name of the Sonnet module
        """
        super().__init__(name=name)

        self._connectivity_radius = connectivity_radius
        self._step_size = step_size
        self._graph_network = graph_model_global.EncodeProcessDecode(output_size=num_dimensions, **graph_network_kwargs)
        self._XLO = system_size[0, :]
        self._XHI = system_size[1, :]
        self._LX = self._XHI - self._XLO
        self._counter = 0
        self._acceleration = acceleration_init
        self._force_per_edge = None
        self._energies = None

    def __call__(self, graph):
        """Apply one step of molecular dynamics

        :param graph: a GraphsTuple having, for some integers N, E, G, Z:
                - nodes: N x Z Tensor of [x_x, x_y, v_x1, v_y1, v_x2, v_x2, mass] for each atom (node). The second to
                    last and third to least columns should represent the velocity (x and y component) of the system at
                    the previous timestep
                - edges: E x Z Tensor of [Z_1, Z_2, ..., Z_Z] for Z features which parameterize the force potential
        :return: A tensor with shape N x 4 of [x_x, x_y, v_x, v_y] that describes the state of the system after 1 time
                step
        """
        next_position, updated_velocities = self.verlet_integrator(graph)
        updated_positions = utils_md.remap_pbc(self._XLO, self._XHI, next_position)
        return tf.concat([updated_positions, updated_velocities], axis=-1)

    def _encoder_preprocessor(self, graph, next_position, s):
        """Prepare data to be fed to the graph network

        The node attributes are a history of C velocities from time t=t to t=t-c with the most recent velocities
        along the right most columns. The edge features have potential parameters and the displacements from the
        sender to the receiver. Edges with displacements greater than the connectivity radius are removed. I added
        code such that bonds are represented as two directional edges (rather than one). The receiver sends a
        negative distance to the sender. I'm not sure if this helps and can simply be removed or commented out.

        :param graph: see initialization function
        :param next_position: Tensor N x 2 of the positions of each atom at time t=t+1
        :return: updated graph of node features and edges to be fed into the graph network.
        """
        edges_conn = tf.stack([graph.senders, graph.receivers], axis=-1)
        edges = graph.edges

        node_features = [next_position]
        r = tf.gather(next_position, edges_conn[:, 1])

        # edge features

        # 1. compute distances
        displacements = r - s # noqa

        # 2. update based on closest periodic image
        displacements = displacements - self._LX * tf.math.rint(displacements / self._LX)

        # 3. Compute norm
        norm = tf.norm(displacements, axis=1, keepdims=True)

        # 4. construct edges
        globals = gn.blocks.broadcast_globals_to_edges(graph)
        edge_features = tf.concat([edges, norm, globals], axis=1)


        # add global features here #
        # currently no global features

        return gn.graphs.GraphsTuple(
            nodes=tf.cast(tf.concat(node_features, axis=-1), dtype=tf.float64),
            edges=tf.cast(edge_features, dtype=tf.float64),
            globals=graph.globals,
            n_node=graph.n_node,
            n_edge=graph.n_edge,
            senders=edges_conn[:, 0],
            receivers=edges_conn[:, 1]
        )

    def _decoder_postprocessor(self, graph, next_position):  # noqa
        """Second part of Verlet integrator to update velocities (from half-velocities)

        :param half_velocity: Tensor N x 2 of velocities for each atom at time step t + 1/2
        :return: Tensor N x 2 of velocities for each atom at time step t + 1
        """
        # Verlet integrator - update positions and half velocity
        return None

    def verlet_integrator(self, graph):
        half_velocity = graph.nodes[:, -3:-1] + 0.5 * self._acceleration * self._step_size
        next_position = graph.nodes[:, :2] + half_velocity * self._step_size

        with tf.GradientTape() as tape:
            s = tf.gather(next_position, graph.senders)
            s = tf.Variable(tf.identity(s))
            input_graphs_tuple = self._encoder_preprocessor(graph, next_position, s)
            energies = self._graph_network(input_graphs_tuple)
            tot_energies = tf.reduce_sum(energies)

        force_per_edge = tape.gradient(tot_energies, s) * -1
        self._force_per_edge = force_per_edge
        self._energies = energies
        sn = input_graphs_tuple.senders
        force_per_node = tf.math.unsorted_segment_sum(force_per_edge, sn, next_position.shape[0])
        self._acceleration = force_per_node
        updated_velocities = half_velocity + 0.5 * self._step_size * force_per_node
        return next_position, updated_velocities

    def get_predicted_accelerations(self,  graph):
        half_velocity = graph.nodes[:, -3:-1] + 0.5 * self._acceleration * self._step_size
        next_position = graph.nodes[:, :2] + half_velocity * self._step_size

        input_graphs_tuple = self._encoder_preprocessor(graph, next_position)
        force = self._graph_network(input_graphs_tuple)

        return force

    def get_step_energy(self,  graph, next_position):
        """Get the accelerations for each atom at a specific position a_t = a_t(x_t)

        For Verlet integration, a(t + dt) is a function of x(t + dt) (positions must be updated first)
        For Euler integration a(t+dt) is a function of x(t) (positions are updated after computing forces)

        :param graph: GraphsTuple of the current state of the system, only the edges are used for computation
        :param next_position: Tensor of positions with shape [num_atoms x 2]
        :return: Tensor of accelerations with shape [num_atoms x2]
        """
        s = tf.gather(next_position, graph.senders)
        input_graphs_tuple = self._encoder_preprocessor(graph, next_position, s)
        energy = self._graph_network(input_graphs_tuple)
        self._energy = energy

        return energy

    def get_step_accelerations(self, graph):
        next_position = graph.nodes[:, :2]
        with tf.GradientTape() as tape:
            s = tf.gather(next_position, graph.senders)
            s = tf.Variable(tf.identity(s))
            input_graphs_tuple = self._encoder_preprocessor(graph, next_position, s)
            energies = self._graph_network(input_graphs_tuple)
            tot_energies = tf.math.reduce_sum(energies)

        force_per_edge = tape.gradient(tot_energies, s) * -1
        self._force_per_edge = force_per_edge
        self._energies = energies
        sn = graph.senders
        force_per_node = tf.math.unsorted_segment_sum(force_per_edge, sn, graph.nodes.shape[0])
        return force_per_node

