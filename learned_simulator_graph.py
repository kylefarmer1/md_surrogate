import graph_model
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
        self._graph_network = graph_model.EncodeProcessDecode(output_size=num_dimensions, **graph_network_kwargs)
        self._XLO = system_size[0, :]
        self._XHI = system_size[1, :]
        self._LX = self._XHI - self._XLO
        self._counter = 0
        self._acceleration = acceleration_init

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
        half_velocity = graph.nodes[:, -3:-1] + 0.5 * self._acceleration * self._step_size
        next_position = graph.nodes[:, :2] + half_velocity * self._step_size

        input_graphs_tuple = self._encoder_preprocessor(graph, next_position)
        assert input_graphs_tuple.globals is None
        force = self._graph_network(input_graphs_tuple)
        self._acceleration = force

        next_velocity = self._decoder_postprocessor(half_velocity)
        updated_positions = utils_md.remap_pbc(self._XLO, self._XHI, next_position)
        return tf.concat([updated_positions, next_velocity], axis=-1)

    def _encoder_preprocessor(self, graph, next_position):
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
        most_recent_position = next_position
        velocity_sequence = graph.nodes[..., 2:-1]
        edges_conn = tf.stack([graph.senders, graph.receivers], axis=-1)
        edges = graph.edges

        # node_features = [velocity_sequence]
        node_features = [next_position]

        # edge features

        # 1. compute distances
        displacements = tf.gather(most_recent_position, edges_conn[:, 1]) - \
                        tf.gather(most_recent_position, edges_conn[:, 0]) # noqa

        # 2. update based on closest periodic image
        displacements = displacements - self._LX * tf.math.rint(displacements / self._LX)

        # 3. mask distances greater than rc
        norm = tf.norm(displacements, axis=1, keepdims=True)
        mask = (norm < self._connectivity_radius)
        mask = mask[:, 0]
        # mask = tf.squeeze(mask)
        # print('#########################################')
        # print(mask)

        # 4. apply mask to edges and edges_conn
        updated_norm = norm[mask]
        updated_edges = edges[mask]
        updated_edge_conn = edges_conn[mask]
        updated_displacements = displacements[mask]
        # relative_displacements = updated_displacements / self._connectivity_radius
        edge_features = tf.concat([updated_edges, updated_displacements, updated_norm], axis=1)

        # 5. update n_edge (wrap in a function probably)
        n_edge = utils_md.compute_n_edge(updated_edge_conn[:, 0], graph.n_node)

        # 6. make edges go both ways (with negative displacements...should they be negative?)
        updated_edge_conn = tf.concat([updated_edge_conn,
                                       tf.gather(updated_edge_conn, [1, 0], axis=1)], axis=0)

        updated_edges_c = tf.concat([edge_features[:, :3], edge_features[:, 3:]*-1], axis=1)
        updated_edges_c = tf.concat([edge_features, updated_edges_c], axis=0)

        mask = tf.argsort(updated_edge_conn[:, 0])
        new_edge_conn = tf.gather(updated_edge_conn, mask)
        new_edges = tf.gather(updated_edges_c, mask)

        n_edge = n_edge * 2

        # add global features here #
        # currently no global features

        return gn.graphs.GraphsTuple(
            nodes=tf.concat(node_features, axis=-1),
            edges=new_edges,
            globals=None,
            n_node=graph.n_node,
            n_edge=n_edge,
            senders=new_edge_conn[:, 0],
            receivers=new_edge_conn[:, 1]
        )

    def _decoder_postprocessor(self, half_velocity):  # noqa
        """Second part of Verlet integrator to update velocities (from half-velocities)

        :param half_velocity: Tensor N x 2 of velocities for each atom at time step t + 1/2
        :return: Tensor N x 2 of velocities for each atom at time step t + 1
        """
        # Verlet integrator - update positions and half velocity
        next_velocity = half_velocity + 0.5 * self._acceleration * self._step_size
        return next_velocity

    def get_predicted_accelerations(self,  graph):
        half_velocity = graph.nodes[:, -3:-1] + 0.5 * self._acceleration * self._step_size
        next_position = graph.nodes[:, :2] + half_velocity * self._step_size

        input_graphs_tuple = self._encoder_preprocessor(graph, next_position)
        force = self._graph_network(input_graphs_tuple)

        return force

    def get_step_accelerations(self,  graph, next_position):
        """Get the accelerations for each atom at a specific position a_t = a_t(x_t)

        For Verlet integration, a(t + dt) is a function of x(t + dt) (positions must be updated first)
        For Euler integration a(t+dt) is a function of x(t) (positions are updated after computing forces)

        :param graph: GraphsTuple of the current state of the system, only the edges are used for computation
        :param next_position: Tensor of positions with shape [num_atoms x 2]
        :return: Tensor of accelerations with shape [num_atoms x2]
        """
        input_graphs_tuple = self._encoder_preprocessor(graph, next_position)
        force = self._graph_network(input_graphs_tuple)

        return force
