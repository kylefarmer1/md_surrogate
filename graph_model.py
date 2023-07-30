import graph_nets as gn
import sonnet as snt
import tensorflow as tf
from typing import Callable


Reducer = Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]


def build_mlp(hidden_size: int, num_hidden_layers: int, output_size: int) -> snt.Module:
    return snt.nets.MLP(
        output_sizes=[hidden_size] * num_hidden_layers + [output_size]
    )


class EncodeProcessDecode(snt.Module): # noqa
    def __init__(
            self,
            latent_size: int,
            mlp_hidden_size: int,
            mlp_num_hidden_layers: int,
            num_message_passing_steps: int,
            output_size: int,
            reducer: Reducer = tf.math.unsorted_segment_sum,
            name: str = "EncodeProcessDecode"):
        super(EncodeProcessDecode, self).__init__(name=name)
        self._latent_size = latent_size
        self._mlp_hidden_size = mlp_hidden_size
        self._mlp_num_hidden_layers = mlp_num_hidden_layers
        self._num_message_passing_steps = num_message_passing_steps
        self._output_size = output_size
        self._reducer = reducer
        self._networks_builder()

    def __call__(self, input_graph: gn.graphs.GraphsTuple) -> tf.Tensor:
        latent_graph_0 = self._encode(input_graph)

        latent_graph_m = self._process(latent_graph_0)

        return self._decode(latent_graph_m)

    def _networks_builder(self):
        def build_mlp_with_layer_norm():
            mlp = build_mlp(
                hidden_size=self._mlp_hidden_size,
                num_hidden_layers=self._mlp_num_hidden_layers,
                output_size=self._latent_size
            )
            return snt.Sequential([mlp, snt.LayerNorm(axis=-1, create_offset=True, create_scale=True)])

        encoder_kwargs = dict(
            edge_model_fn=build_mlp_with_layer_norm,
            node_model_fn=build_mlp_with_layer_norm
        )
        self._encoder_network = gn.modules.GraphIndependent(**encoder_kwargs)

        self._processor_networks = []
        for _ in range(self._num_message_passing_steps):
            self._processor_networks.append(
                gn.modules.InteractionNetwork(
                    edge_model_fn=build_mlp_with_layer_norm,
                    node_model_fn=build_mlp_with_layer_norm,
                    reducer=self._reducer
                )
            )

        self._decoder_network = build_mlp(
            hidden_size=self._mlp_hidden_size,
            num_hidden_layers=self._mlp_num_hidden_layers,
            output_size=self._output_size
        )

    def _encode(self, input_graph: gn.graphs.GraphsTuple) -> gn.graphs.GraphsTuple:

        if input_graph.globals is not None:
            broadcasted_globals = gn.blocks.broadcast_globals_to_nodes(input_graph)
            input_graph = input_graph.replace(
                nodes=tf.concat([input_graph.nodes, broadcasted_globals], axis=-1),
                globals=None
            )

        latent_graph_0 = self._encoder_network(input_graph)
        return latent_graph_0

    def _process(
            self, latent_graph_0: gn.graphs.GraphsTuple) -> gn.graphs.GraphsTuple:

        latent_graph_prev_k = latent_graph_0
        latent_graph_k = latent_graph_0
        for processor_network_k in self._processor_networks:
            latent_graph_k = self._process_step(
                processor_network_k, latent_graph_prev_k
            )
            latent_graph_prev_k = latent_graph_k

        latent_graph_m = latent_graph_k
        return latent_graph_m

    # noinspection PyMethodMayBeStatic
    def _process_step(
            self, processor_network_k: snt.Module, latent_graph_prev_k: gn.graphs.GraphsTuple) -> gn.graphs.GraphsTuple:

        latent_graph_k = processor_network_k(latent_graph_prev_k) # noqa

        latent_graph_k = latent_graph_k.replace(
            nodes=latent_graph_k.nodes+latent_graph_prev_k.nodes,
            edges=latent_graph_k.edges+latent_graph_prev_k.edges
        )
        return latent_graph_k

    def _decode(self, latent_graph: gn.graphs.GraphsTuple) -> tf.Tensor:
        return self._decoder_network(latent_graph.nodes) # noqa # noqa
