import os
import common
import argparse
import numpy as np

class Scale:
    """
    Scales a bunch of meshes.
    """

    def __init__(self):
        """
        Constructor.
        """

        parser = self.get_parser()
        self.options = parser.parse_args()

    def get_parser(self):
        """
        Get parser of tool.

        :return: parser
        """

        parser = argparse.ArgumentParser(description='Scale a set of meshes stored as OFF files.')
        parser.add_argument('--in_dir', type=str, help='Path to input directory.')
        parser.add_argument('--out_dir', type=str, help='Path to output directory; files within are overwritten!')
        parser.add_argument('--padding', type=float, default=0.1, help='Relative padding applied on each side.')
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

    def run(self):
        """
        Run the tool, i.e. scale all found OFF files.
        """

        assert os.path.exists(self.options.in_dir)
        common.makedir(self.options.out_dir)
        files = self.read_directory(self.options.in_dir)

        for filepath in files:
            mesh = common.Mesh.from_off(filepath)

            # Get extents of model.
            min, max = mesh.extents()
            total_min = np.min(np.array(min))
            total_max = np.max(np.array(max))

            # Set the center (although this should usually be the origin already).
            centers = (
                (min[0] + max[0]) / 2,
                (min[1] + max[1]) / 2,
                (min[2] + max[2]) / 2
            )
            # Scales all dimensions equally.
            sizes = (
                total_max - total_min,
                total_max - total_min,
                total_max - total_min
            )
            translation = (
                -centers[0],
                -centers[1],
                -centers[2]
            )
            scales = (
                1 / (sizes[0] + 2 * self.options.padding * sizes[0]),
                1 / (sizes[1] + 2 * self.options.padding * sizes[1]),
                1 / (sizes[2] + 2 * self.options.padding * sizes[2])
            )

            mesh.translate(translation)
            mesh.scale(scales)

            print('[Data] %s extents before %f - %f, %f - %f, %f - %f' % (os.path.basename(filepath), min[0], max[0], min[1], max[1], min[2], max[2]))
            min, max = mesh.extents()
            print('[Data] %s extents after %f - %f, %f - %f, %f - %f' % (os.path.basename(filepath), min[0], max[0], min[1], max[1], min[2], max[2]))

            # May also switch axes if necessary.
            #mesh.switch_axes(1, 2)

            mesh.to_off(os.path.join(self.options.out_dir, os.path.basename(filepath)))

if __name__ == '__main__':
    app = Scale()
    app.run()