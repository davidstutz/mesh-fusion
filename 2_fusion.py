import math
import numpy as np
import os
from scipy import ndimage
import common
import argparse
import ntpath

# Import shipped libraries.
import librender
import libmcubes

use_gpu = True
if use_gpu:
    import libfusiongpu as libfusion
    from libfusiongpu import tsdf_gpu as compute_tsdf
else:
    import libfusioncpu as libfusion
    from libfusioncpu import tsdf_cpu as compute_tsdf

class Fusion:
    """
    Performs TSDF fusion.
    """

    def __init__(self):
        """
        Constructor.
        """

        parser = self.get_parser()
        self.options = parser.parse_args()

        self.render_intrinsics = np.array([
            self.options.focal_length_x,
            self.options.focal_length_y,
            self.options.principal_point_x,
            self.options.principal_point_x
        ], dtype=float)
        # Essentially the same as above, just a slightly different format.
        self.fusion_intrisics = np.array([
            [self.options.focal_length_x, 0, self.options.principal_point_x],
            [0, self.options.focal_length_y, self.options.principal_point_y],
            [0, 0, 1]
        ])
        self.image_size = np.array([
            self.options.image_height,
            self.options.image_width,
        ], dtype=np.int32)
        # Mesh will be centered at (0, 0, 1)!
        self.znf = np.array([
            1 - 0.75,
            1 + 0.75
        ], dtype=float)
        # Derive voxel size from resolution.
        self.voxel_size = 1./self.options.resolution
        self.truncation = self.options.truncation_factor*self.voxel_size

    def get_parser(self):
        """
        Get parser of tool.

        :return: parser
        """

        parser = argparse.ArgumentParser(description='Scale a set of meshes stored as OFF files.')
        parser.add_argument('--mode', type=str, default='render', help='Operation mode: render or fuse.')
        parser.add_argument('--in_dir', type=str, help='Path to input directory.')
        parser.add_argument('--depth_dir', type=str, help='Path to depth directory; files are overwritten!')
        parser.add_argument('--out_dir', type=str, help='Path to output directory; files within are overwritten!')
        parser.add_argument('--n_views', type=int, default=100, help='Number of views per model.')
        parser.add_argument('--image_height', type=int, default=640, help='Depth image height.')
        parser.add_argument('--image_width', type=int, default=640, help='Depth image width.')
        parser.add_argument('--focal_length_x', type=float, default=640, help='Focal length in x direction.')
        parser.add_argument('--focal_length_y', type=float, default=640, help='Focal length in y direction.')
        parser.add_argument('--principal_point_x', type=float, default=320, help='Principal point location in x direction.')
        parser.add_argument('--principal_point_y', type=float, default=320, help='Principal point location in y direction.')

        parser.add_argument('--depth_offset_factor', type=float, default=1.5, help='The depth maps are offsetted using depth_offset_factor*voxel_size.')
        parser.add_argument('--resolution', type=float, default=256, help='Resolution for fusion.')
        parser.add_argument('--truncation_factor', type=float, default=10, help='Truncation for fusion is derived as truncation_factor*voxel_size.')

        return parser

    def read_directory(self, directory):
        """
        Read directory.

        :param directory: path to directory
        :return: list of files
        """

        files = []
        for filename in os.listdir(directory):
            files.append(os.path.normpath(os.path.join(directory, filename)))

        return files

    def get_points(self):
        """
        See https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere.

        :param n_points: number of points
        :type n_points: int
        :return: list of points
        :rtype: numpy.ndarray
        """

        rnd = 1.
        points = []
        offset = 2. / self.options.n_views
        increment = math.pi * (3. - math.sqrt(5.));

        for i in range(self.options.n_views):
            y = ((i * offset) - 1) + (offset / 2);
            r = math.sqrt(1 - pow(y, 2))

            phi = ((i + rnd) % self.options.n_views) * increment

            x = math.cos(phi) * r
            z = math.sin(phi) * r

            points.append([x, y, z])

        # visualization.plot_point_cloud(np.array(points))
        return np.array(points)

    def get_views(self):
        """
        Generate a set of views to generate depth maps from.

        :param n_views: number of views per axis
        :type n_views: int
        :return: rotation matrices
        :rtype: [numpy.ndarray]
        """

        Rs = []
        points = self.get_points()

        for i in range(points.shape[0]):
            # https://math.stackexchange.com/questions/1465611/given-a-point-on-a-sphere-how-do-i-find-the-angles-needed-to-point-at-its-ce
            longitude = - math.atan2(points[i, 0], points[i, 1])
            latitude = math.atan2(points[i, 2], math.sqrt(points[i, 0] ** 2 + points[i, 1] ** 2))

            R_x = np.array([[1, 0, 0], [0, math.cos(latitude), -math.sin(latitude)], [0, math.sin(latitude), math.cos(latitude)]])
            R_y = np.array([[math.cos(longitude), 0, math.sin(longitude)], [0, 1, 0], [-math.sin(longitude), 0, math.cos(longitude)]])

            R = R_y.dot(R_x)
            Rs.append(R)

        return Rs

    def render(self, mesh, Rs):
        """
        Render the given mesh using the generated views.

        :param base_mesh: mesh to render
        :type base_mesh: mesh.Mesh
        :param Rs: rotation matrices
        :type Rs: [numpy.ndarray]
        :return: depth maps
        :rtype: numpy.ndarray
        """

        depthmaps = []
        for i in range(len(Rs)):
            np_vertices = Rs[i].dot(mesh.vertices.astype(np.float64).T)
            np_vertices[2, :] += 1

            np_faces = mesh.faces.astype(np.float64)
            np_faces += 1

            depthmap, mask, img = librender.render(np_vertices.copy(), np_faces.T.copy(), self.render_intrinsics, self.znf, self.image_size)

            # This is mainly result of experimenting.
            # The core idea is that the volume of the object is enlarged slightly
            # (by subtracting a constant from the depth map).
            # Dilation additionally enlarges thin structures (e.g. for chairs).
            depthmap -= self.options.depth_offset_factor * self.voxel_size
            depthmap = ndimage.morphology.grey_erosion(depthmap, size=(3, 3))

            depthmaps.append(depthmap)

        return depthmaps

    def fusion(self, depthmaps, Rs):
        """
        Fuse the rendered depth maps.

        :param depthmaps: depth maps
        :type depthmaps: numpy.ndarray
        :param Rs: rotation matrices corresponding to views
        :type Rs: [numpy.ndarray]
        :return: (T)SDF
        :rtype: numpy.ndarray
        """

        Ks = self.fusion_intrisics.reshape((1, 3, 3))
        Ks = np.repeat(Ks, len(depthmaps), axis=0).astype(np.float32)

        Ts = []
        for i in range(len(Rs)):
            Rs[i] = Rs[i]
            Ts.append(np.array([0, 0, 1]))

        Ts = np.array(Ts).astype(np.float32)
        Rs = np.array(Rs).astype(np.float32)

        depthmaps = np.array(depthmaps).astype(np.float32)
        views = libfusion.PyViews(depthmaps, Ks, Rs, Ts)

        # Note that this is an alias defined as libfusiongpu.tsdf_gpu or libfusioncpu.tsdf_cpu!
        return compute_tsdf(views, self.options.resolution, self.options.resolution, self.options.resolution,
                            self.voxel_size, self.truncation, False)

    def run(self):
        """
        Run the tool.
        """

        if self.options.mode == 'render':
            self.run_render()
        elif self.options.mode == 'fuse':
            self.run_fuse()
        else:
            print('Invalid model, choose render or fuse.')
            exit()

    def run_render(self):
        """
        Run rendering.
        """

        assert os.path.exists(self.options.in_dir)
        common.makedir(self.options.depth_dir)

        files = self.read_directory(self.options.in_dir)
        timer = common.Timer()
        Rs = self.get_views()

        for filepath in files:
            timer.reset()
            mesh = common.Mesh.from_off(filepath)
            depths = self.render(mesh, Rs)

            depth_file = os.path.join(self.options.depth_dir, os.path.basename(filepath) + '.h5')
            common.write_hdf5(depth_file, np.array(depths))
            print('[Data] wrote %s (%f seconds)' % (depth_file, timer.elapsed()))

    def run_fuse(self):
        """
        Run fusion.
        """

        assert os.path.exists(self.options.depth_dir)
        common.makedir(self.options.out_dir)

        files = self.read_directory(self.options.depth_dir)
        timer = common.Timer()
        Rs = self.get_views()

        for filepath in files:

            # As rendering might be slower, we wait for rendering to finish.
            # This allows to run rendering and fusing in parallel (more or less).

            depths = common.read_hdf5(filepath)

            timer.reset()
            tsdf = self.fusion(depths, Rs)
            tsdf = tsdf[0]

            vertices, triangles = libmcubes.marching_cubes(-tsdf, 0)
            vertices /= self.options.resolution
            vertices -= 0.5

            off_file = os.path.join(self.options.out_dir, ntpath.basename(filepath)[:-3])
            libmcubes.export_off(vertices, triangles, off_file)
            print('[Data] wrote %s (%f seconds)' % (off_file, timer.elapsed()))

if __name__ == '__main__':
    app = Fusion()
    app.run()