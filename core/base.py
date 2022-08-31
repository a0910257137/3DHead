import tensorflow as tf
import numpy as np
from pytorch3d.renderer import (look_at_view_transform, FoVPerspectiveCameras,
                                PointLights, RasterizationSettings,
                                MeshRenderer, MeshRasterizer, SoftPhongShader,
                                TexturesVertex, blending)


class Base:
    def __init__(self, config, **kwargs):
        self.config = config
        self.batch_size = self.config.batch_size

    def _get_reverse_z(self):
        return tf.reshape(tf.constant([1.0, 0, 0, 0, 1, 0, 0, 0, -1.0]),
                          [1, 3, 3])

    def _get_camera_pose(self):
        return tf.reshape(tf.constant([0.0, 0.0, 10.0]), (1, 1, 3))

    def _get_p_mat(self, img_size, focal):
        half_image_width = img_size // 2
        p_matrix = tf.reshape(
            tf.constant([
                focal, 0.0, half_image_width, 0.0, focal, half_image_width,
                0.0, 0.0, 1.0
            ]), [1, 3, 3])
        return p_matrix

    def _get_renderer(self, img_size, focal, device):

        R, T = look_at_view_transform(10, 0, 0)  # camera's position
        cameras = FoVPerspectiveCameras(
            device=device,
            R=R,
            T=T,
            znear=0.01,
            zfar=50,
            fov=2 * np.arctan(img_size // 2 / focal) * 180. / np.pi)

        lights = PointLights(device=device,
                             location=[[0.0, 0.0, 1e5]],
                             ambient_color=[[1, 1, 1]],
                             specular_color=[[0., 0., 0.]],
                             diffuse_color=[[0., 0., 0.]])

        raster_settings = RasterizationSettings(
            image_size=img_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        blend_params = blending.BlendParams(background_color=[0, 0, 0])

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras,
                                      raster_settings=raster_settings),
            shader=SoftPhongShader(device=device,
                                   cameras=cameras,
                                   lights=lights,
                                   blend_params=blend_params))
        return renderer
