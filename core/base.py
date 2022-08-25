import tensorflow as tf
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (look_at_view_transform, FoVPerspectiveCameras,
                                PointLights, RasterizationSettings,
                                MeshRenderer, MeshRasterizer, SoftPhongShader,
                                TexturesVertex, blending)



class Base:

    def __init__(self, config):
        self.config = config

        self.batch = self.config.batch_size
        self.batch_size = self.config.batch_size
        self.focal = self.config.focal
        self.img_size = self.config.img_size[0]
        self.renderer = self._get_renderer()

        self.reverse_z = self._get_reverse_z()
        self.camera_pos = self._get_camera_pose()
        self.p_mat = self._get_p_mat()
        self.id_tensor = None
        self.exp_tensor = None
        self.tex_tensor = None
        self.rot_tensor = None
        self.gamma_tensor = None
        self.trans_tensor = None
        self.id_dims = 80
        self.tex_dims = 80
        self.exp_dims = 64
        self.init_coeff_tensors()

    def _packed_tensors(self):
        return tf.concat([
            tf.tile(self.id_tensor, [self.batch, 1]), self.exp_tensor,
            tf.tile(self.tex_tensor, [self.batch, 1]), self.rot_tensor,
            self.gamma_tensor, self.trans_tensor
        ],
                         axis=-1)

    def _split_coeffs(self, coeffs):
        id_coeff = coeffs[:, :80]  # identity(shape) coeff of dim 80
        exp_coeff = coeffs[:, 80:144]  # expression coeff of dim 64
        tex_coeff = coeffs[:, 144:224]  # texture(albedo) coeff of dim 80
        # ruler angles(x,y,z) for rotation of dim 3
        angles = coeffs[:, 224:227]
        # lighting coeff for 3 channel SH function of dim 27
        gamma = coeffs[:, 227:254]
        translation = coeffs[:, 254:]  # translation coeff of dim 3
        return id_coeff, exp_coeff, tex_coeff, angles, gamma, translation

    def init_coeff_tensors(self, id_coeff=None, tex_coeff=None):
        if id_coeff is None:
            self.id_tensor = tf.zeros((1, self.id_dims), dtype=tf.float32)
        else:
            assert id_coeff.shape == (1, self.id_dims)

        if tex_coeff is None:
            self.tex_tensor = tf.zeros((1, self.tex_dims), dtype=tf.float32)

        else:
            assert tex_coeff.shape == (1, self.tex_dims)
            self.tex_tensor = tf.constant(tex_coeff, dtype=tf.float32)
        self.exp_tensor = tf.zeros((self.batch_size, self.exp_dims),
                                   dtype=tf.float32)
        self.gamma_tensor = tf.zeros((self.batch_size, 27), dtype=tf.float32)
        self.trans_tensor = tf.zeros((self.batch_size, 3), dtype=tf.float32)
        self.rot_tensor = tf.zeros((self.batch_size, 3), dtype=tf.float32)

    def get_trans_tensor(self):
        return self.trans_tensor

    def get_exp_tensor(self):
        return self.exp_tensor

    def get_tex_tensor(self):
        return self.tex_tensor

    def get_id_tensor(self):
        return self.id_tensor

    def get_gamma_tensor(self):
        return self.gamma_tensor

    def get_skinmask(self):
        return self.skinmask


    def _get_reverse_z(self):
        return tf.reshape(tf.constant([1.0, 0, 0, 0, 1, 0, 0, 0, -1.0]),
                          [1, 3, 3])

    def _get_camera_pose(self):
        return tf.reshape(tf.constant([0.0, 0.0, 10.0]), (1, 1, 3))

    def _get_p_mat(self):
        half_image_width = self.img_size // 2
        p_matrix = tf.reshape(
            tf.constant([
                self.focal, 0.0, half_image_width, 0.0, self.focal,
                half_image_width, 0.0, 0.0, 1.0
            ]), [1, 3, 3])
        return p_matrix

    def _get_renderer(self):
        R, T = look_at_view_transform(10, 0, 0)  # camera's position
        cameras = FoVPerspectiveCameras(
            device='cpu',
            R=R,
            T=T,
            znear=0.01,
            zfar=50,
            fov=2 * np.arctan(self.img_size // 2 / self.focal) * 180. / np.pi)

        lights = PointLights(device='cpu',
                             location=[[0.0, 0.0, 1e5]],
                             ambient_color=[[1, 1, 1]],
                             specular_color=[[0., 0., 0.]],
                             diffuse_color=[[0., 0., 0.]])

        raster_settings = RasterizationSettings(
            image_size=self.img_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        blend_params = blending.BlendParams(background_color=[0, 0, 0])

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras,
                                      raster_settings=raster_settings),
            shader=SoftPhongShader(device='cpu',
                                   cameras=cameras,
                                   lights=lights,
                                   blend_params=blend_params))
        return renderer

    