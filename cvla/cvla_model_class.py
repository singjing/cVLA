from scipy.spatial.transform import Rotation as R
import numpy as np
from PIL import Image
from mani_skill.utils.structs import Pose
from mani_skill.examples.utils_trajectory import unproject_points
from copy import deepcopy
from cvla.utils_trajectory import DummyCamera
from matplotlib import pyplot as plt
from pathlib import Path
from cvla.hf_model_class import cVLA_wrapped
import torch
from torchvision.transforms import v2
from cvla.utils_traj_tokens import getActionEncInstance
import json

class CVLA:
    def __init__(self, model_path="/home/houman/cVLA_test/models", dataset_path="/home/houman/cVLA_test/"):
        # some processing
        self.model_location = Path(model_path)
        self.model_path = self.model_location / "mix30obj_mask" / "checkpoint-4687"
        self.model = cVLA_wrapped(model_path=self.model_path)
        self.dataset_location = Path(dataset_path)
        self.camera_extrinsic = [[[1, 0, 0.0, 0.0], [0, 1, 0, 0], [0, 0, 1, 0]]]
        self.camera_intrinsic = [[[260.78692626953125, 0.0, 322.3820495605469],[ 0.0, 260.78692626953125, 180.76370239257812],[0.0, 0.0, 1.0]]]

        self.info_file = self.model_path.parent / "cvla_info.json"
        try:
            with open(self.info_file, "r") as f:
                self.model_info = json.load(f)
        except FileNotFoundError:
            self.model_info = None

        if self.model_info is not None:
            self.action_encoder = self.model_info["action_encoder"]
            self.return_depth = self.model_info["return_depth"]
        else:
            self.action_encoder = "xyzrotvec-cam-1024xy"
            self.return_depth = False
            if "_depth" in str(model_path):
                self.return_depth = True

        self.enc_model = getActionEncInstance(self.action_encoder)
        self.dataset_name = self.dataset_location.name
        self.model_name = self.model_path.parent.name
        print("dataset:".ljust(10), self.dataset_name, self.dataset_location)
        if self.model_path.is_dir():
            print("model:".ljust(10), self.model_name,"\t", self.model_path)
            print("encoder".ljust(10), self.action_encoder)
            print("depth:".ljust(10), self.return_depth)


    
    def find_closest_valid_pixel(self, depth_image, target_row, target_col):
        """
        Find the closest valid pixel value to the target position in a depth image.
        
        Args:
            depth_image: 2D numpy array representing the depth image
            target_row: Row index of the position to find closest valid pixel for
            target_col: Column index of the position to find closest valid pixel for
        
        Returns:
            float: Value of the closest valid pixel, or None if no valid pixels exist
        """
        rows, cols = depth_image.shape
        print (rows, cols)
        if target_row >= rows or target_col >= cols:
            raise ValueError("Target coordinates exceed image dimensions")
        closest_pixel_coords = None
        # Get coordinates of all valid pixels
        valid_coords = [(i,j) for i in range(rows) 
                        for j in range(cols) 
                        if not np.isnan(depth_image[i,j])]
        
        # If target position already has a valid value, return it
        if not np.isnan(depth_image[target_row, target_col]):
            print("The coordinate already has a Non-Nan value!")
            return depth_image[target_row, target_col], [target_row, target_col]
        
        # If no valid pixels exist, return None
        if not valid_coords:
            return None, None
        
        # Find closest valid pixel using Manhattan distance
        min_dist = float('inf')
        closest_value = None
        
        for row, col in valid_coords:
            dist = abs(target_row - row) + abs(target_col - col)
            if dist < min_dist:
                min_dist = dist
                closest_value = depth_image[row, col]
                closest_pixel_coords = [row,col]
                
        return closest_value, closest_pixel_coords



    def get_object_pose(self, rgb_image, depth_image, string_x="yellow cup", string_y="blue bowl"):

        action_text = "put the {first} inside the {second}".format(first = string_x, second = string_y)

        image_width_no_crop, image_height_no_crop = rgb_image.size
        print("original image size", image_width_no_crop, image_height_no_crop)

        camera_extrinsic = [[[1, 0, 0.0, 0.0], [0, 1, 0, 0], [0, 0, 1, 0]]]
        camera_intrinsic = [[[260.78692626953125, 0.0, 322.3820495605469],[ 0.0, 260.78692626953125, 180.76370239257812],[0.0, 0.0, 1.0]]]
        camera_no_crop = DummyCamera(camera_intrinsic, camera_extrinsic, width=image_width_no_crop, height=image_height_no_crop)

        crop = True
        if crop:
            center_crop = v2.CenterCrop(360)
            image = center_crop(rgb_image)
            depth_image = center_crop(depth_image)
        else:
            image = rgb_image

        image_width, image_height = image.size
        print("new image size", image_width, image_height)

        # compute intrinsic matrix for cropped camera
        dx = int((image_width_no_crop - image_width) / 2)
        dy = int((image_height_no_crop - image_height) / 2)
        K = np.array(camera_intrinsic[0])  # shape (3,3)
        K_cropped = K.copy()
        K_cropped[0,2] -= dx
        K_cropped[1,2] -= dy
        camera = DummyCamera([K_cropped.tolist()], camera_extrinsic, width=image_width, height=image_height)

        base_to_tcp_pos = torch.tensor([[[-0.7487 + .7, -0.3278 + 0.3 ,  0.7750]]])
        base_to_tcp_orn = torch.tensor([[[ 1,  0, 0, 0]]])  # quaternion w, x, y, z 
        _, _, robot_state = self.model.enc_model.encode_trajectory(base_to_tcp_pos, base_to_tcp_orn, camera)

        if self.model.return_depth:
            images = [rgb_image, depth_image]
            print(images[0].shape, images[0].dtype)  # (720, 1280, 3) uint8 depth image encoded
            print(images[1])  # <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1280x720 at 0x715AEF7667B0>
        else:
            images = rgb_image
            if crop:
                images = center_crop(images)

        pred_text = self.model.predict(images, action_text, robot_state)

        # decc_model = self.model.enc_model.decode_caption
        # sample = dict(suffix=pred_text[0], prefix=action_text + " " + robot_state)
        # image_crop_width, image_crop_height = images.size
        # camera_after_crop = DummyCamera(camera_intrinsic, camera_extrinsic, width=image_crop_width, height=image_crop_height)
        traj_c = self.model.enc_model.decode_caption(pred_text[0], camera=camera)
        traj_c_fixed = deepcopy(traj_c)
        points = [] 
        for point_idx in (0,1):
            point = np.array(traj_c[0][point_idx,:2].round().numpy(), dtype=int)
            row, col = point[1], point[0]  # Position containing NaN
            closest_value, closest_pixel_coords = self.find_closest_valid_pixel(depth_image, row, col)
            print(f"Closest valid value to position ({row}, {col}): {closest_value} at ({closest_pixel_coords[0]}, {closest_pixel_coords[1]})")
            depth_value = closest_value #depth_image[point[1], point[0]]

            print("Point, Depth value: ", point, depth_value)
            traj_c_fixed[0][point_idx,2] = float(depth_value)
            points.append(point)
        print("old", traj_c[0])
        print("fixed", traj_c_fixed[0])
        curve_25d, quat_c = traj_c_fixed
        # from camera to world coordinates
        extrinsic_orn = R.from_matrix(camera.get_extrinsic_matrix()[:, :3, :3])
        extrinsic = Pose.create_from_pq(p=camera.get_extrinsic_matrix()[:, :3, 3],
                                        q=extrinsic_orn.as_quat(scalar_first=True))
        quat_w = extrinsic.inv() * Pose.create_from_pq(q=quat_c)
        curve_w = unproject_points(camera, curve_25d) 

        curve_w, quat_w.get_q().unsqueeze(0)  # shape (P, 3 = u, v, d)    
        print(curve_w)
        print(quat_w)

        return curve_w, quat_w.raw_pose.numpy(), points, images
            



# if __name__ == "__main__":
#     from data_loader_jsonl import JSONLDataset

#     # define paths
#     dataset_location = "/data/lmbraid19/argusm/datasets/clevr-real-block-simple-v3"
#     model_location = Path("/data/lmbraid19/argusm/models/")
#     model_path = model_location / "clevr-act-7-depth_depthaug" / "checkpoint-4687"    

#     # load model
#     model_inst = cVLA_wrapped(model_path)

#     # load some test data
#     return_depth = model_inst.return_depth
#     test_dataset = JSONLDataset(
#         jsonl_file_path=f"{dataset_location}/_annotations.valid.jsonl",
#         image_directory_path=f"{dataset_location}/dataset",
#         clean_prompt=True,
#         return_depth=return_depth
#     )
#     test_dataset.action_encoder = getActionEncInstance("xyzrotvec-cam-1024xy")

#     print("dataset len", len(test_dataset))
#     print(model_path)

#     # access some images
#     images, entry = test_dataset[0]
#     action_text = entry["prefix"].split("<")[0].strip()

#     print(action_text)  # e.g. put the yellow block in the blue cup
#     print(images[0].shape, images[0].dtype)  # (720, 1280, 3) uint8 depth image encoded
#     print(images[1])  # <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1280x720 at 0x715AEF7667B0>

#     # camera_extrinsic = [[[1, 0, 0.0, 0.0], [0, 1, 0, 0], [0, 0, 1, 0]]]
#     # camera_intrinsic = [[[260.78692626953125, 0.0, 322.3820495605469],[ 0.0, 260.78692626953125, 180.76370239257812],[0.0, 0.0, 1.0]]]
#     # camera = DummyCamera(camera_intrinsic, camera_extrinsic, width=image_width, height=image_height)

#     # make a prediction
#     traj = model_inst.predict_trajectory(images, action_text=action_text, camera=entry["camera"])
#     print(traj)

