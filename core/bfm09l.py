from .base import Base
import tensorflow as tf
import numpy as np
import math
from pprint import pprint
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (look_at_view_transform, FoVPerspectiveCameras,
                                PointLights, RasterizationSettings,
                                MeshRenderer, MeshRasterizer, SoftPhongShader,
                                TexturesVertex, blending)
class BFM09(Base):

    def __init__(self, config, model_dict, **kargs):

        Base.__init__(self, config, **kargs)
        self.config = config
        self.skinmask = tf.cast(model_dict['skinmask'], tf.float32)

        self.kp_inds = tf.cast(model_dict['keypoints'].squeeze(), tf.int32)

        self.meanshape = tf.cast(model_dict['meanshape'], tf.float32)

        self.idBase = tf.cast(model_dict['idBase'], tf.float32)

        self.expBase = tf.cast(model_dict['exBase'], tf.float32)
        self.meantex = tf.cast(model_dict['meantex'], tf.float32)
        self.texBase = tf.cast(model_dict['texBase'], tf.float32)

        self.tri = tf.cast(model_dict['tri'] - 1, tf.int64)
        self.point_buf = tf.cast(model_dict['point_buf'] - 1, tf.int64)
        self.keys = ["lms_proj"]
        if self.config.render:
            self.keys += ["rendered_img", "face_texture", "vs", "tri", "color"]

        

    def __call__(self, coeffs, render=True):

        batch = tf.shape(coeffs)[0]
        id_coeff, exp_coeff, tex_coeff, angles, gamma, translation = self._split_coeffs(
            coeffs)
        vs = self.get_vs(batch, id_coeff, exp_coeff)  # [1, 35709, 3]
        rotation = self.compute_rotation_matrix(batch, angles)  # [1, 3, 3]
        vs_t = self.rigid_transform(vs, rotation, translation)  # [1, 35709, 3]
        # 3D  to 2D transformation
        lms_t = self.get_lms(batch, vs_t)
        lms_proj = self.project_vs(batch, lms_t)

        lms_proj = tf.stack(
            [lms_proj[:, :, 0], self.img_size - lms_proj[:, :, 1]], axis =2)
        if render:
            face_texture = self.get_color(batch, tex_coeff)
            face_norm = self.compute_norm(batch, vs, self.tri, self.point_buf)
            face_norm_r = tf.linalg.matmul(face_norm, rotation)
            face_color = self.add_illumination(face_texture, face_norm_r,
                                               gamma)

            tf.numpy_function(self._render_mesh, 
                                inp=[vs_t, face_color, 
                                tf.tile(self.tri[None, ...], [batch, 1, 1])], 
                            Tout=[tf.float32])
            face_color_tv = TexturesVertex(face_color)
            mesh = Meshes(vs_t, self.tri.repeat(batch, 1, 1),
                          face_color_tv)
            rendered_img = self.renderer(mesh)
            rendered_img = tf.clip_by_value(rendered_img, 0., 255.)
            return {
                'rendered_img': rendered_img,
                'lms_proj': lms_proj,
                'face_texture': face_texture,
                'vs': vs_t,
                'tri': self.tri,
                'color': face_color
                }
        else:
            return {'lms_proj': lms_proj}
        

    def get_vs(self, batch, id_coeff, exp_coeff):
        face_shape = tf.einsum('ij,aj->ai', self.idBase, id_coeff) + \
            tf.einsum('ij,aj->ai', self.expBase, exp_coeff) + self.meanshape
        face_shape = tf.reshape(face_shape, [batch, -1, 3])
        face_shape = face_shape - tf.math.reduce_mean(
            tf.reshape(self.meanshape, [1, -1, 3]), axis=1, keepdims=True)
        return face_shape

    def compute_rotation_matrix(self, batch, angles):
        sinx = math.sin(angles[:, 0])
        siny = math.sin(angles[:, 1])
        sinz = math.sin(angles[:, 2])
        cosx = math.cos(angles[:, 0])
        cosy = math.cos(angles[:, 1])
        cosz = math.cos(angles[:, 2])

        rot_matrix = [[[1., 0., 0.], [0., cosx, -sinx], [0., sinx, cosx]],
                      [[cosy, 0., siny], [0., 1., 0.], [-siny, 0., cosy]],
                      [[cosz, -sinz, 0.], [sinz, cosz, 0.], [0., 0., 1.]]]
        rotXYZ = tf.constant(rot_matrix, shape=(3, 1, 3, 3))
        rotXYZ = tf.tile(rotXYZ, [1, batch, 1, 1])

        rotation = tf.linalg.matmul(tf.linalg.matmul(rotXYZ[0], rotXYZ[1]),
                                    rotXYZ[2])
        rotation = tf.transpose(rotation, [0, 2, 1])
        return rotation

    def rigid_transform(self, vs, rot, trans):
        vs_r = tf.matmul(vs, rot)
        vs_t = vs_r + tf.reshape(trans, [-1, 1, 3])
        return vs_t

    def get_lms(self, batch, vs):
        b_idx = tf.range(batch)[:, None]
        n = self.kp_inds.get_shape().as_list()
        b_idx = tf.tile(b_idx, [n[0], 1])
        inds = tf.concat([ b_idx, self.kp_inds[:, None]], axis = -1)
        lms = tf.reshape(tf.gather_nd(vs, inds), [batch, n[0], 3])
        return lms
        
    def get_color(self, batch , tex_coeff):
        face_texture = tf.einsum('ij,aj->ai', self.texBase,  
                                    tex_coeff) + self.meantex
        face_texture = tf.reshape(face_texture, [batch, -1, 3])
        return face_texture

    def project_vs(self, batch, vs):
        vs  =  tf.matmul(vs, tf.tile(self.reverse_z, [batch, 1, 1]))  + self.camera_pos
        aug_projection = tf.matmul(vs, tf.transpose(tf.tile(self.p_mat, [batch, 1, 1]), [0, 2, 1]))
        face_projection = aug_projection[:, :, :2] / \
            tf.reshape(aug_projection[:, :, 2], [batch, -1, 1])
        return face_projection

    def compute_norm(self, batch, vs, tri, point_buf):
        face_id = tf.cast(tri, tf.int32)
        n, _ = face_id.get_shape().as_list()
        point_id = point_buf
        b_idx = tf.tile(tf.range(batch, dtype=tf.int32)[:, None], [n, 1])
        v1 = tf.reshape(tf.gather_nd(vs, tf.concat([b_idx, face_id[:, :1]], axis =- 1)), (-1, n, 3))
        v2 = tf.reshape(tf.gather_nd(vs,  tf.concat([b_idx, face_id[:, 1:2]], axis =- 1)),  (-1, n, 3))
        
        v3 = tf.reshape(tf.gather_nd(vs,   tf.concat([b_idx, face_id[:, 2:]], axis =- 1)), (-1, n, 3))
        e1 = v1 - v2
        e2 = v2 - v3
        face_norm = tf.linalg.cross(e1, e2)

        empty = tf.zeros((tf.shape(face_norm)[0], 1, 3),
                            dtype=face_norm.dtype)

        face_norm = tf.concat((face_norm, empty), 1)
        v_norm = tf.numpy_function(self.get_nnorm, inp=[ face_norm, point_id], Tout=[tf.float32])
        return v_norm

    def get_nnorm(self, face_norm, point_id):
        v_norm = np.sum(face_norm [:, point_id, :], axis = 2)
        v_norm = v_norm / np.expand_dims(np.linalg.norm(v_norm, axis = 2), axis = -1)
        return  v_norm

    def add_illumination(self, face_texture, norm, gamma):
        n_b, num_vertex, _ = face_texture.get_shape().as_list()
        n_v_full = n_b * num_vertex
        gamma = tf.identity(tf.reshape(gamma, (-1, 3, 9)))
        gamma = tf.concat([gamma[..., :1] + 0.8, gamma[..., 1:]], axis = -1)
        gamma = tf.transpose(gamma, [0, 2, 1])

        a0 = np.pi
        a1 = 2 * np.pi / np.sqrt(3.0)
        a2 = 2 * np.pi / np.sqrt(8.0)
        c0 = 1 / np.sqrt(4 * np.pi)
        c1 = np.sqrt(3.0) / np.sqrt(4 * np.pi)
        c2 = 3 * np.sqrt(5.0) / np.sqrt(12 * np.pi)
        d0 = 0.5 / np.sqrt(3.0)

        Y0 = tf.ones(n_v_full) * a0 * c0

        norm = tf.reshape(norm, [-1, 3])

        nx, ny, nz = norm[:, 0], norm[:, 1], norm[:, 2]
        arrH = []

        arrH.append(Y0)
        arrH.append(-a1 * c1 * ny)
        arrH.append(a1 * c1 * nz)
        arrH.append(-a1 * c1 * nx)
        arrH.append(a2 * c2 * nx * ny)
        arrH.append(-a2 * c2 * ny * nz)
        arrH.append(a2 * c2 * d0 * (3 *  tf.math.pow(nz, 2) - 1))
        arrH.append(-a2 * c2 * nx * nz)
        arrH.append(a2 * c2 * 0.5 * (tf.math.pow(nx, 2) - tf.math.pow(ny, 2)))

        H = tf.stack(arrH, 1)
        Y = tf.reshape(H, [n_b, num_vertex, 9])
        lighting = tf.linalg.matmul(Y, gamma)
        face_color = face_texture * lighting

        return face_color

    def _render_mesh(self, vs_t, face_color, tri):
        vs_t = torch.tensor(vs_t, 
                            dtype=torch.float32,
                            requires_grad=False,
                            device='cpu')

        face_color = torch.tensor(face_color, 
                                dtype=torch.float32,
                                requires_grad=False,
                                device='cpu')
        tri = torch.tensor(tri, 
                        dtype=torch.float32,
                        requires_grad=False, 
                        device='cpu')
    
        
        face_color_tv = TexturesVertex(face_color)
        mesh = Meshes(vs_t, tri, face_color_tv)
        
        rendered_img = self.renderer(mesh)
        
        # rendered_img = torch.clamp(rendered_img, 0, 255)
        
        
        return 