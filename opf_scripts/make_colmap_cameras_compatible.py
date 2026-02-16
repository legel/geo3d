'''
Opens the colmap cameras file and changes the format from Full_OpenCV to OpenCV, which
is needed for nerfstudio.
'''

import json
import pathlib

import argparse

def make_colmap_cameras_compatible(cameras_path: pathlib.Path, output_path: pathlib.Path) -> None:
    """
    Make a camera file that is compatible with nerfstudio.

    From the OPF the cameras file is of the following format:

    cameras.txt
    ```txt
    # Camera list with one line of data per camera:
    #   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
    #   FULL_OPENCV model: CAMERA_ID FULL_OPENCV WIDTH HEIGHT fx fy cx cy k1 k2 p1 p2 k3 k4 k5 k6
    # Number of cameras: 3
    1 FULL_OPENCV 5280 3956 3716.1379634705486 3716.1379634705486 2627.7909343926726 1925.531579643719 -0.10841462700837586 0.0033175741653546586 2.8716084987264094e-06 -7.955413792490326e-05 -0.019128156099607264 0.0 0.0 0.0
    2 FULL_OPENCV 4224 2376 2800.6853747491177 2800.6853747491177 2132.989062410066 1185.9892300680767 0.16751758001629213 -0.43694085113871817 0.0 0.0 0.3572620582476267 0.0 0.0 0.0
    3 FULL_OPENCV 5280 3956 3711.4736242481904 3711.4736242481904 2630.6611799725556 1927.3369352072907 0.0068451060448535975 -0.01720111628858226 -0.001854310406489115 -0.00021452167090679883 0.035920615364993885 0.0 0.0 0.0
    ```

    We want to use OPENCV instead of FULL_OPENCV, which is the format that nerfstudio expects.

    cameras_out.txt
    ```txt
    # Camera list with one line of data per camera:
    #   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
    #   OPENCV model: CAMERA_ID OPENCV WIDTH HEIGHT fx fy cx cy k1 k2 p1 p2 k3 k4
    # Number of cameras: 3
    1 OPENCV 5280 3956 3716.1379634705486 3716.1379634705486 2627.7909343926726 1925.531579643719 -0.10841462700837586 0.0033175741653546586 2.8716084987264094e-06 -7.955413792490326e-05 -0.019128156099607264 0.0
    2 OPENCV 4224 2376 2800.6853747491177 2800.6853747491177 2132.989062410066 1185.9892300680767 0.16751758001629213 -0.43694085113871817 0.0 0.0 0.3572620582476267 0.0 0.0 0.0
    3 OPENCV 5280 3956 3711.4736242481904 3711.4736242481904 2630.6611799725556 1927.3369352072907 0.0068451060448535975 -0.01720111628858226 -0.001854310406489115 -0.00021452167090679883 0.035920615364993885 0.0
    ```

    Args:
        cameras_path (pathlib.Path): Path to the cameras.txt file.
        output_path (pathlib.Path): Path to save the modified cameras.txt file.
    """
    with open(cameras_path, "r") as f:
        lines = f.readlines()

    # Modify the camera lines
    for i in range(1, len(lines)):

        # First modify the preamble lines with "#"
        if lines[i].startswith("#"):
            lines[i] = lines[i].replace("FULL_OPENCV", "OPENCV")
            lines[i] = lines[i].replace("k5 k6", "k5")
            continue

        # Replace FULL_OPENCV with OPENCV
        lines[i] = lines[i].replace("FULL_OPENCV", "OPENCV")

        # Then modify the actual camera data lines to remove k5 and k6
        parts = lines[i].strip().split()
        num_params_per_line_OPENCV = 16  # OPENCV has 16 parameters
        if len(parts) > num_params_per_line_OPENCV:
            parts = parts[:num_params_per_line_OPENCV]
            lines[i] = " ".join(parts) + "\n"

    # Write the modified lines to the output file
    with open(output_path, "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make COLMAP cameras compatible with nerfstudio.")
    parser.add_argument("cameras_path", type=pathlib.Path, help="Path to the cameras.txt file.")
    parser.add_argument("output_path", type=pathlib.Path, help="Path to save the modified cameras.txt file.")

    args = parser.parse_args()
    make_colmap_cameras_compatible(args.cameras_path, args.output_path)
    print(f"Modified cameras file saved to {args.output_path}")
    
    print("Done.")
    print("The cameras file has been modified to be compatible with nerfstudio.")
    print("You can now use this file with nerfstudio or other compatible tools.")
