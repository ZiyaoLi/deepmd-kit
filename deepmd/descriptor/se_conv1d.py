import math
import numpy as np
from typing import Tuple, List

from deepmd.env import tf
from deepmd.common import get_activation_func, get_precision, ACTIVATION_FN_DICT, PRECISION_DICT, docstring_parameter, \
    get_np_precision
from deepmd.utils.argcheck import list_to_doc
from deepmd.env import GLOBAL_TF_FLOAT_PRECISION
from deepmd.env import GLOBAL_NP_FLOAT_PRECISION
from deepmd.env import op_module
from deepmd.env import default_tf_session_config
from deepmd.utils.network import embedding_net, conv1d_net
from deepmd.utils.tabulate import DeepTabulate
from deepmd.utils.type_embed import embed_atom_type_from_atype

from .se_a import DescrptSeA


class DescrptSeConv1d(DescrptSeA):
    @docstring_parameter(list_to_doc(ACTIVATION_FN_DICT.keys()), list_to_doc(PRECISION_DICT.keys()))
    def __init__(self,
                 conv_windows=(3, 3, 3),
                 conv_neurons=(96, 96, 96),
                 conv_residual=False,
                 **kwargs
                 ) -> None:
        """
        Constructor

        Parameters
        ----------
        conv_windows: list of int
            The window sizes of conv1d kernels

        conv_neurons: list of int
            The number of output channels of conv1d kernels

        kwargs
            all other parameters for DescrptSeA
        """
        super(DescrptSeConv1d, self).__init__(**kwargs)
        self.conv_windows = conv_windows
        self.conv_neurons = conv_neurons
        self.conv_residual = conv_residual

    def get_dim_out(self) -> int:
        """
        Returns the output dimension of this descriptor
        """
        return self.conv_neurons[-1] if len(self.conv_neurons) > 0 else self.get_dim_before_conv()

    def get_dim_before_conv(self) -> int:
        """
        Returns the output dimension of this descriptor before convolution
        The same as DescrptSeA.get_dim_out()
        """
        return self.filter_neuron[-1] * self.n_axis_neuron

    def _pass_filter(self,
                     inputs,
                     atype,
                     natoms,
                     input_dict,
                     reuse=None,
                     suffix='',
                     trainable=True):
        try:
            type_embedding = input_dict['type_embedding']
        except TypeError:
            raise ValueError("must provide the input dict with type embedding in it.")
        except KeyError:
            raise ValueError("must provide type embedding in the input dict.")

        inputs = tf.reshape(inputs, [-1, self.ndescrpt * natoms[0]])
        inputs_i = tf.reshape(inputs, [-1, self.ndescrpt])
        dout, qmat = self._filter_(tf.cast(inputs_i, self.filter_precision),
                                   atype=atype,
                                   name='filter_type_all' + suffix,
                                   natoms=natoms,
                                   reuse=reuse,
                                   seed=self.seed,
                                   trainable=trainable,
                                   activation_fn=self.filter_activation_fn,
                                   type_embedding=type_embedding)
        dout = tf.reshape(dout, [tf.shape(inputs)[0], natoms[0], self.get_dim_before_conv()])
        dout = conv1d_net(dout, self.conv_windows, self.conv_neurons,
                          name='conv1d_descrpt', residual=self.conv_residual)
        dout = tf.reshape(dout, [tf.shape(inputs)[0], natoms[0] * self.get_dim_out()])
        qmat = tf.reshape(qmat, [tf.shape(inputs)[0], natoms[0] * self.get_dim_rot_mat_1() * 3])

        return dout, qmat

    def _filter_(
            self,
            inputs,
            atype,
            natoms,
            type_embedding = None,
            activation_fn=tf.nn.tanh,
            stddev=1.0,
            bavg=0.0,
            name='linear',
            reuse=None,
            seed=None,
            trainable = True):

        nframes = tf.shape(tf.reshape(inputs, [-1, natoms[0], self.ndescrpt]))[0]

        # natom x (nnei x 4)
        shape = inputs.get_shape().as_list()  # = [None, self.ndescrpt]

        outputs_size = self.filter_neuron[-1]
        outputs_size_2 = self.n_axis_neuron

        with tf.variable_scope(name, reuse=reuse):
            xyz_scatter_1 = self._filter_lower_(
                inputs,
                atype,
                nframes,
                natoms,
                type_embedding=type_embedding,
                is_exclude=False,
                activation_fn=activation_fn,
                stddev=stddev,
                bavg=bavg,
                seed=seed,
                trainable=trainable)

            # natom x nei x outputs_size
            # xyz_scatter = tf.concat(xyz_scatter_total, axis=1)
            # natom x nei x 4
            # inputs_reshape = tf.reshape(inputs, [-1, shape[1]//4, 4])
            # natom x 4 x outputs_size
            # xyz_scatter_1 = tf.matmul(inputs_reshape, xyz_scatter, transpose_a = True)
            xyz_scatter_1 = xyz_scatter_1 * (4.0 / shape[1])
            # natom x 4 x outputs_size_2
            xyz_scatter_2 = tf.slice(xyz_scatter_1, [0,0,0],[-1,-1,outputs_size_2])
            # # natom x 3 x outputs_size_2
            # qmat = tf.slice(xyz_scatter_2, [0,1,0], [-1, 3, -1])
            # natom x 3 x outputs_size_1
            qmat = tf.slice(xyz_scatter_1, [0,1,0], [-1, 3, -1])
            # natom x outputs_size_1 x 3
            qmat = tf.transpose(qmat, perm = [0, 2, 1])
            # natom x outputs_size x outputs_size_2
            result = tf.matmul(xyz_scatter_1, xyz_scatter_2, transpose_a = True)
            # natom x (outputs_size x outputs_size_2)
            result = tf.reshape(result, [-1, outputs_size * outputs_size_2])

        return result, qmat

    def _filter_lower_(
            self,
            inputs,
            atype,
            nframes,
            natoms,
            type_embedding=None,
            is_exclude=False,
            activation_fn=None,
            bavg=0.0,
            stddev=1.0,
            seed=None,
            trainable=True,
            suffix='',
    ):
        """
        input env matrix, returns R.G
        """
        outputs_size = [1] + self.filter_neuron
        inputs_i = inputs  # (nf x na, nnei x 4)
        shape_i = inputs_i.get_shape().as_list()  # = [None, self.ndescrpt]
        inputs_reshape = tf.reshape(inputs_i, [-1, 4])  # (nf x na x nnei, 4)
        # with (natom x nei_type_i) x 1
        xyz_scatter = tf.reshape(       # get the first dim, (nf x na x nnei, 1)
            tf.slice(inputs_reshape, [0, 0], [-1, 1]), [-1, 1])

        if type_embedding is not None:
            type_embedding = tf.cast(type_embedding, self.filter_precision)
            xyz_scatter = self._concat_type_embedding_(
                xyz_scatter, atype, nframes, natoms, type_embedding)
            if self.compress:
                raise NotImplementedError('compression of type embedded descriptor is not supported at the moment.')
        # with (natom x nei_type_i) x out_size
        if self.compress and (not is_exclude):
            raise NotImplementedError('compression of type embedded descriptor is not supported at the moment.')
        else:
            if not is_exclude:
                xyz_scatter = embedding_net(
                    xyz_scatter,
                    self.filter_neuron,
                    self.filter_precision,
                    activation_fn=activation_fn,
                    resnet_dt=self.filter_resnet_dt,
                    name_suffix=suffix,
                    stddev=stddev,
                    bavg=bavg,
                    seed=seed,
                    trainable=trainable)
            else:
                w = tf.zeros((outputs_size[0], outputs_size[-1]), dtype=GLOBAL_TF_FLOAT_PRECISION)
                xyz_scatter = tf.matmul(xyz_scatter, w)
            # natom x nei_type_i x out_size
            xyz_scatter = tf.reshape(xyz_scatter, (-1, shape_i[1] // 4, outputs_size[-1]))
            return tf.matmul(tf.reshape(inputs_i, [-1, shape_i[1] // 4, 4]), xyz_scatter, transpose_a=True)

    def _concat_type_embedding_(
            self,
            xyz_scatter,
            atype,
            nframes,
            natoms,
            type_embedding,
    ):
        te_out_dim = type_embedding.get_shape().as_list()[-1]
        nei_embed = tf.nn.embedding_lookup(type_embedding, tf.cast(self.nei_type, dtype=tf.int32))  # nnei*nchnl
        nei_embed = tf.tile(nei_embed, (nframes * natoms[0], 1))
        nei_embed = tf.reshape(nei_embed, [-1, te_out_dim])
        embedding_input = tf.concat([xyz_scatter, nei_embed], 1)
        if not self.type_one_side:
            atm_embed = embed_atom_type_from_atype(atype, type_embedding)
            atm_embed = tf.tile(atm_embed, (1, self.nnei))
            atm_embed = tf.reshape(atm_embed, [-1, te_out_dim])
            embedding_input = tf.concat([embedding_input, atm_embed], 1)
        return embedding_input

