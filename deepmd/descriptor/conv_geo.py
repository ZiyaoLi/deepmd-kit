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

from .se_conv1d import DescrptSeConv1d


class DescrptSeConvGeo(DescrptSeConv1d):
    DIM_GEOM_FEATS = 5  # d, cos phi, sin phi, cos psi, sin psi

    @docstring_parameter(list_to_doc(ACTIVATION_FN_DICT.keys()), list_to_doc(PRECISION_DICT.keys()))
    def __init__(self,
                 conv_geo_windows: list = [7, 3, 3],
                 conv_geo_neurons: list = [100, 100, 100],
                 conv_geo_residual: bool = True,
                 conv_geo_activation_fn: str = 'tanh',
                 **kwargs
                 ) -> None:
        self.conv_geo_windows = conv_geo_windows
        self.conv_geo_neurons = conv_geo_neurons
        self.conv_geo_residual = conv_geo_residual
        try:
            self.conv_geo_activation_fn = ACTIVATION_FN_DICT[conv_geo_activation_fn]
        except KeyError:
            raise ValueError("unknown activation function type: %s" % conv_geo_activation_fn)
        super(DescrptSeConvGeo, self).__init__(**kwargs)

    def get_dim_conv1d(self):
        return super(DescrptSeConvGeo, self).get_dim_out()

    def get_dim_out(self) -> int:
        """
        Returns the output dimension of this descriptor
        """
        return self.get_dim_conv1d() + DescrptSeConvGeo.DIM_GEOM_FEATS + \
            self.conv_geo_neurons[-1] if len(self.conv_geo_neurons) > 0 else 0

    def build(self,
              coord_: tf.Tensor,
              atype_: tf.Tensor,
              natoms: tf.Tensor,
              box_: tf.Tensor,
              mesh: tf.Tensor,
              input_dict: dict,
              reuse: bool = None,
              suffix: str = ''
              ) -> tf.Tensor:
        """
        see DescrptSeA.build for explanations.
        """
        davg = self.davg
        dstd = self.dstd
        with tf.variable_scope('descrpt_attr' + suffix, reuse=reuse):
            if davg is None:
                davg = np.zeros([self.ntypes, self.ndescrpt])
            if dstd is None:
                dstd = np.ones([self.ntypes, self.ndescrpt])
            t_rcut = tf.constant(np.max([self.rcut_r, self.rcut_a]),
                                 name='rcut',
                                 dtype=GLOBAL_TF_FLOAT_PRECISION)
            t_ntypes = tf.constant(self.ntypes,
                                   name='ntypes',
                                   dtype=tf.int32)
            t_ndescrpt = tf.constant(self.ndescrpt,
                                     name='ndescrpt',
                                     dtype=tf.int32)
            t_sel = tf.constant(self.sel_a,
                                name='sel',
                                dtype=tf.int32)
            self.t_avg = tf.get_variable('t_avg',
                                         davg.shape,
                                         dtype=GLOBAL_TF_FLOAT_PRECISION,
                                         trainable=False,
                                         initializer=tf.constant_initializer(davg))
            self.t_std = tf.get_variable('t_std',
                                         dstd.shape,
                                         dtype=GLOBAL_TF_FLOAT_PRECISION,
                                         trainable=False,
                                         initializer=tf.constant_initializer(dstd))

        coord = tf.reshape(coord_, [-1, natoms[1] * 3])
        box = tf.reshape(box_, [-1, 9])
        atype = tf.reshape(atype_, [-1, natoms[1]])

        self.descrpt, self.descrpt_deriv, self.rij, self.nlist \
            = op_module.prod_env_mat_a(coord,
                                       atype,
                                       natoms,
                                       box,
                                       mesh,
                                       self.t_avg,
                                       self.t_std,
                                       rcut_a=self.rcut_a,
                                       rcut_r=self.rcut_r,
                                       rcut_r_smth=self.rcut_r_smth,
                                       sel_a=self.sel_a,
                                       sel_r=self.sel_r)
        # only used when tensorboard was set as true
        tf.summary.histogram('descrpt', self.descrpt)
        tf.summary.histogram('rij', self.rij)
        tf.summary.histogram('nlist', self.nlist)

        self.descrpt_reshape = tf.reshape(self.descrpt, [-1, self.ndescrpt])
        self.descrpt_reshape = tf.identity(self.descrpt_reshape, name='o_rmat')
        self.descrpt_deriv = tf.identity(self.descrpt_deriv, name='o_rmat_deriv')
        self.rij = tf.identity(self.rij, name='o_rij')
        self.nlist = tf.identity(self.nlist, name='o_nlist')

        dout, self.qmat = self._pass_filter(self.descrpt_reshape,
                                            atype,
                                            natoms,
                                            input_dict,
                                            suffix=suffix,
                                            reuse=reuse,
                                            trainable=self.trainable)

        dout = tf.reshape(dout, [tf.shape(dout)[0], natoms[0], self.get_dim_conv1d()])
        geom_feats = self.build_local_geometries(tf.reshape(coord, [tf.shape(coord)[0], natoms[0], 3]))
        dout = tf.concat([dout, geom_feats], -1, name='full_descrpt_with_geom')
        if len(self.conv_geo_windows) > 0:
            conv_geo_out = conv1d_net(geom_feats, self.conv_geo_windows, self.conv_geo_neurons,
                                      name='conv_geo_descrpt',
                                      activation_fn=self.conv_geo_activation_fn,
                                      residual=self.conv_geo_residual)
            dout = tf.concat([dout, conv_geo_out], -1, name='full_descrpt_with_conved_geom')

        self.dout = dout
        # only used when tensorboard was set as true
        tf.summary.histogram('embedding_net_output', self.dout)

        return self.dout

    def build_local_geometries(self, coord: tf.Tensor):

        def inner_product(x, y, axis=-1):
            return tf.reduce_sum(x * y, axis=axis)

        def get_angle_normal_vec(vec_to_prev, vec_to_next):
            a1 = vec_to_prev[:, :, 0]
            a2 = vec_to_prev[:, :, 1]
            a3 = vec_to_prev[:, :, 2]
            b1 = vec_to_next[:, :, 0]
            b2 = vec_to_next[:, :, 1]
            b3 = vec_to_next[:, :, 2]
            n1 = a2 * b3 - a3 * b2
            n2 = a3 * b1 - a1 * b3
            n3 = a1 * b2 - a2 * b1
            nvec = tf.stack([n1, n2, n3], axis=-1)
            nvec = tf.math.divide_no_nan(nvec, tf.norm(nvec, ord=2, axis=-1, keepdims=True))
            return nvec

        with tf.variable_scope("get_local_geometry"):
            # calc d
            vec_to_prev_ = coord[:, 1:, :] - coord[:, :-1, :]  # (1, 2, ..., L)
            vec_to_next_ = coord[:, :-1, :] - coord[:, 1:, :]  # (0, 1, ..., L-1)
            dist = tf.norm(vec_to_prev_, ord=2, axis=-1)  # [(0, 1), (1, 2), ..., (L-1, L)]
            # padding the vectors to (1, 2, ..., L-1)
            vec_to_prev = vec_to_prev_[:, :-1, :]
            vec_to_next = vec_to_next_[:, 1:, :]

            # calc phi
            vec_inner_prod = inner_product(vec_to_prev, vec_to_next)
            cos_phi = tf.math.divide_no_nan(vec_inner_prod, dist[:, :-1] * dist[:, 1:])
            sin_phi = tf.sqrt(1 - cos_phi * cos_phi)

            # calc psi
            nvec = get_angle_normal_vec(vec_to_prev, vec_to_next)
            cos_psi = inner_product(nvec[:, :-1, :], nvec[:, 1:, :])
            sin_psi = tf.sqrt(1 - cos_psi * cos_psi)

        # calc geom feats
        geom_feats = self.pad_and_stack_geom_feats(dist, cos_phi, sin_phi, cos_psi, sin_psi)

        return geom_feats

    def pad_and_stack_geom_feats(self, dist_, cos_phi_, sin_phi_, cos_psi_, sin_psi_):
        # create padding variables
        with tf.variable_scope("local_geometry_padding"):
            nframe = tf.shape(dist_)[0]
            pad_dist    = tf.broadcast_to(tf.Variable(1., name="pad_distance", dtype=self.filter_precision),
                                          (nframe, 1))
            pad_cos_phi = tf.broadcast_to(tf.Variable([-.5, -.5], name="pad_cos_phi", dtype=self.filter_precision),
                                          (nframe, 2))
            pad_sin_phi = tf.sqrt(1 - pad_cos_phi * pad_cos_phi, name="pad_sin_phi")
            pad_cos_psi = tf.broadcast_to(tf.Variable([-.5, -.5, -.5], name="pad_cos_psi", dtype=self.filter_precision),
                                          (nframe, 3))
            pad_sin_psi = tf.sqrt(1 - pad_cos_psi * pad_cos_psi, name="pad_sin_psi")

            # padding the results.
            dist = tf.concat([dist_, pad_dist], -1)
            cos_phi = tf.concat([cos_phi_, pad_cos_phi], -1)
            sin_phi = tf.concat([sin_phi_, pad_sin_phi], -1)
            cos_psi = tf.concat([cos_psi_, pad_cos_psi], -1)
            sin_psi = tf.concat([sin_psi_, pad_sin_psi], -1)
            return tf.stack([dist, cos_phi, sin_phi, cos_psi, sin_psi], -1)




