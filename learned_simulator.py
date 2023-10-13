import graph_model
import graph_nets as gn
import sonnet as snt
import tensorflow as tf
import connectivity_utils
import utils_md


class LearnedSimulator(snt.Module):  # noqa

    def __init__(
            self,
            num_dimensions,
            step_size,
            system_size,
            connectivity_radius,
            graph_network_kwargs,
            name="LearnedSimulator"):
        super().__init__(name=name)

        self._connectivity_radius = connectivity_radius
        self.step_size = step_size
        self._graph_network = graph_model.EncodeProcessDecode(output_size=num_dimensions, **graph_network_kwargs)
        self._XLO = system_size[0, :]
        self._XHI = system_size[1, :]
        self._LX = self._XHI - self._XLO
        self._counter = 0

    def __call__(self, position_velocity_sequence, n_particles_per_example, edges, edges_conn, global_context=None):
        next_position_t = position_velocity_sequence[:, -1, :2]
        input_graphs_tuple = self._encoder_preprocessor(
            position_velocity_sequence[:, :-1, :], next_position_t, n_particles_per_example,
            edges, edges_conn
        )
        acceleration_t = self._graph_network(input_graphs_tuple)
        next_position, half_velocity = self._decoder_postprocessor_1(
            acceleration_t, position_velocity_sequence)

        input_graphs_tuple = self._encoder_preprocessor(
            position_velocity_sequence[:, 1:, :], next_position, n_particles_per_example, edges, edges_conn)
        acceleration = self._graph_network(input_graphs_tuple)
        next_position, next_velocity = self._decoder_postprocessor_2(
            acceleration, next_position, half_velocity
        )
        return tf.concat([next_position, next_velocity], axis=-1)

    def _encoder_preprocessor(
            self, position_velocity_sequence, next_position, n_node, edges, edges_conn, global_context=None):
        most_recent_position = next_position
        velocity_sequence = position_velocity_sequence[..., 2:4]
        # update position here to fall in line with the verlet integrator

        node_features = []

        # implement MergeDims here
        flat_velocity_sequence = tf.reshape(velocity_sequence, (velocity_sequence.shape[0], -1))
        node_features.append(flat_velocity_sequence)

        # edge features
        edge_features = []
        # 1. compute distances
        displacements = (
                tf.gather(most_recent_position, edges_conn[:, 0]) -
                tf.gather(most_recent_position, edges_conn[:, 1])
        )

        # 2. update based on closest periodic image
        displacements = displacements - self._LX * tf.math.rint(displacements / self._LX)

        # 3. mask distances greater than rc
        mask = (tf.norm(displacements, axis=1) < self._connectivity_radius)

        # 4. apply mask to edges and edges_conn
        updated_edges = edges[mask]
        updated_edge_conn = edges_conn[mask]
        updated_displacements = displacements[mask]
        relative_displacements = updated_displacements / self._connectivity_radius
        edge_features.extend([updated_edges, relative_displacements])

        # 5. update n_edge (wrap in a function probably)
        n_node_cumsum = tf.math.cumsum(n_node)
        split = tf.where(updated_edge_conn[:, 0:1] < n_node_cumsum)
        diff = split[1:, 0] - split[:-1, 0]
        n_edge = tf.cast(tf.where(diff), dtype=tf.int32)
        n_edge = tf.concat([n_edge[:, 0], tf.shape(split[:-1, 1])], axis=0)
        n_edge_r = n_edge[1:] - n_edge[:-1]
        n_edge = tf.concat([n_edge[0:1]+1, n_edge_r], axis=0)
        n_edge = tf.concat([n_edge[0:1], n_edge[1:] - n_edge[:-1]], axis=0)

        # add global features here #
        # currently no global features

        return gn.graphs.GraphsTuple(
            nodes=tf.cast(tf.concat(node_features, axis=-1), tf.float32),
            edges=tf.cast(tf.concat(edge_features, axis=-1), tf.float32),
            globals=global_context,
            n_node=n_node,
            n_edge=n_edge,
            senders=updated_edge_conn[:, 0],
            receivers=updated_edge_conn[:, 1]
        )

    def _decoder_postprocessor_1(self, acceleration, position_velocity_sequence):  # noqa

        # Verlet integrator - update positions and half velocity
        most_recent_position = position_velocity_sequence[:, -1, :2]
        most_recent_velocity = position_velocity_sequence[:, -1, 2:4]

        half_velocity = most_recent_velocity + 0.5 * acceleration * self.step_size
        new_position = most_recent_position + half_velocity * self.step_size
        return new_position, half_velocity

    def _decoder_postprocessor_2(self, acceleration, next_position, half_velocity):  # noqa
        # Verlet integrator
        new_velocity = half_velocity + 0.5 * acceleration * self.step_size  # * dt = 1
        # implement PBC
        new_position = utils_md.remap_pbc(self._XLO, self._XHI, self._LX, next_position)
        return next_position, new_velocity

    def get_predicted_accelerations(
            self,  next_position, position_velocity_sequence, n_particles_per_example, edges, edges_conn):
        # Having the model calculate this acceleration might not be correct. Should be able to back calculate the
        # previous acceleration from the positions and velocities
        # also need to make sure that the target and predicted velocities are aligned correctly
        # basically just make sure all the time steps for pos and vel are correct.
        # Might also be worth it to simplify all these inputs and outputs
        # also might be worth checking to see if not normalizing the distances would be better.
        input_graphs_tuple = self._encoder_preprocessor(
            position_velocity_sequence, next_position, n_particles_per_example, edges, edges_conn)

        predicted_acceleration = self._graph_network(input_graphs_tuple)

        return predicted_acceleration

    def _inverse_decoder_postprocessor(self, next_pos_vel, next_next_pos):
        # need to work out the dimensions here
        acceleration = (next_next_pos - next_pos_vel[..., :2] - next_pos_vel[..., 2:4] * self.step_size) * \
                       2 / self.step_size ** 2
        return acceleration
