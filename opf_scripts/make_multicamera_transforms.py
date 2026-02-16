'''
Open a transforms.json file from pyopf and make a new transforms.json file
that uses the new convention for transforms with multicamera data.
'''

import json
import pathlib

import argparse

def make_multicamera_transforms(transforms_path: pathlib.Path, output_path: pathlib.Path) -> None:
    """
    The transforms.json file from pyopf is of the following format:
    {
        // Preamble with intrinsics of _one_ of the cameras (e.g., camera model 2)
        "camera_model": "OPENCV",
        "camera_angle_x": 1.2922413358739202,
        "camera_angle_y": 0.8023549843539378,
        "fl_x": 2800.6853747491177,
        "fl_y": 2800.6853747491177,
        ...
        "frames": [
            {
                "file_path": "path/to/image_1.png",
                "transform_matrix": [...],
                // Intrinsics for camera 1
                "camera_model": "OPENCV",
                "camera_angle_x": 1.2353634462972731,
                "camera_angle_y": 0.9782629285891327,
                "fl_x": 3716.1379634705486,
                "fl_y": 3716.1379634705486,
                ...
            },
            ...
            {
                "file_path": "path/to/image_235.png",
                "transform_matrix": [...],
                // No intrinsics for camera 2 since in preamble
            },
            ...
            {
                "file_path": "path/to/image_1376.png",
                "transform_matrix": [...],
                // Intrinsics for camera 3
                "camera_model": "OPENCV",
                "camera_angle_x": 1.2365496407798062,
                "camera_angle_y": 0.9793051369679774,
                "fl_x": 3711.4736242481904,
                "fl_y": 3711.4736242481904,
                ...
            },
        ],
        // Epilog with other metadata (e.g., average position and scale)
    }

    What we instead need is to remove the preamble and have the intrinsics
    for each camera in each frame.

    Args:
        transforms_path (pathlib.Path): Path to the transforms.json file.
        output_path (pathlib.Path): Path to save the modified transforms.json file.
    """
    with open(transforms_path, "r") as f:
        transforms = json.load(f)

    # Check multiple frames are in the transforms file
    if 'frames' not in transforms:
        raise ValueError("The transforms.json file does not contain 'frames' key.")
    if not isinstance(transforms['frames'], list) or len(transforms['frames']) == 0:
        raise ValueError("The transforms.json file does not contain any frames.")

    # The preamble will include everything before "frames"
    preamble = {}

    # The epilog will include everything after "frames"
    epilog = {}

    filling_preamble = True

    # Extract the preamble and epilog
    for key, value in transforms.items():
        if key == "frames":
            filling_preamble = False
            continue

        if filling_preamble:
            preamble[key] = value
        else:
            epilog[key] = value

    print("Preamble keys:", preamble.keys())
    print("Epilog keys:", epilog.keys())

    # Get the keys for the camera intrinsics
    intrinsics_keys = preamble.keys()

    # Check which camera intrinsics need to be added to the frames with
    # the missing intrinsics

    for frame in transforms["frames"]:
        # If the frame already has camera intrinsics, skip it
        if all(key in frame for key in intrinsics_keys):
            print(f"Frame {frame['file_path']} already has camera intrinsics, skipping.")
            continue

        print(f"Frame {frame['file_path']} is missing camera intrinsics, adding from preamble.")

        # Add the camera intrinsics from the preamble to the frame
        for key in intrinsics_keys:
            frame[key] = preamble[key]

    # Remove the preamble from the transforms
    transforms = {"frames": transforms["frames"], **epilog}

    # Write the new transforms.json file
    with open(output_path, "w") as f:
        json.dump(transforms, f, indent=4)


if __name__ == "__main__":
    print("Make a multicamera transforms.json file.")

    parser = argparse.ArgumentParser(description="Make a multicamera transforms.json file.")

    print("Arguments:")
    parser.add_argument('transforms_path', type=pathlib.Path, help="Path to the transforms.json file.")
    parser.add_argument('output_path', type=pathlib.Path, help="Path to save the modified transforms.json file.")
    args = parser.parse_args()
    print(f"transforms_path: {args.transforms_path}")
    print(f"output_path: {args.output_path}")
    
    make_multicamera_transforms(args.transforms_path, args.output_path)
    print("Done.")
    print("The transforms.json file has been modified to include camera intrinsics for each frame.")
    print("You can now use this file with nerfstudio or other compatible tools.")
