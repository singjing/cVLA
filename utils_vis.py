import io
import html
import base64
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from utils_traj_tokens import getActionEncInstance
from utils_trajectory import DummyCamera, project_points, convert_to_tensor
from PIL import Image


def get_standard_camera(image_height, image_width):
    camera_extrinsic = [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]]
    camera_intrinsic = [[[410.029, 0.0, 224.0], [0.0, 410.029, 224.0], [0.0, 0.0, 1.0]]]
    camera = DummyCamera(camera_intrinsic, camera_extrinsic, width=image_width, height=image_height)
    return camera

def extract_extrinsics(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Takes a vector [tx ty tz qw qx qy qz]
    return: translation vector (3) and rotation matrix (3, 3)
    """
    translation = pose[:3]
    rotation = Rotation.from_quat(np.array(pose[3:]), scalar_first=True).as_matrix()
    return (translation, rotation)

def local_to_world(t, rot):
    """
    Extrinsic matrix C from local coordinates to world coordinates.
    First apply rotation, then translation.
    t: (3) translation, world coordinates of local center
    rot: (3,3) rotation
    return: (4,4)
    """
    ext = np.eye(4)
    ext[0:3, 0:3] = rot
    ext[0:3, 3] = t
    return ext

def transform_3d(point_3d: np.ndarray, mat: np.ndarray):
    """ 
    Multiply point by transformation matrix in 3D in homogenous coordinates.
    point_3d: (3)
    mat: (4, 4)
    return: (3)
    """
    p3d = np.ones((4))
    p3d[:3] = point_3d # (x y z 1)
    res = mat @ p3d # (4, 4) x (4, 1)
    res = res / res[3] # remove homogeneous coordinate
    return res[:3]

def draw_coordinate_frame(label, camera, enc, ax):
    # === draw axis frame
    # curve shape (B, N, X Y Z) in camera 3D space
    # quat shape (B, N, qw qx qy qz)
    curve_w, quat_w = enc.decode_trajectory(label, camera)

    # coordinate frame
    dist = 0.1
    c = np.array([0, 0, 0])
    x = np.array([dist, 0, 0])
    y = np.array([0, dist, 0])
    z = np.array([0, 0, dist])

    # get transform from TCP coors to camera 3D
    pose = np.array([
        curve_w[0, 0, 0], # x
        curve_w[0, 0, 1], # y
        curve_w[0, 0, 2], # z
        quat_w[0, 0, 0], # qw
        quat_w[0, 0, 1], # qx
        quat_w[0, 0, 2], # qy
        quat_w[0, 0, 3] # qz
    ])
    t, rot = extract_extrinsics(pose) # (3), (3, 3)
    tcp_transform = local_to_world(t, rot) # (4, 4)

    # apply 3D transform
    c = transform_3d(c, tcp_transform) # (xyz)
    x = transform_3d(x, tcp_transform)
    y = transform_3d(y, tcp_transform)
    z = transform_3d(z, tcp_transform)

    # project with intrinsic
    points_3d = np.stack((c, x, y, z))[None, :, :] # (xyz) -> (1, 4, xyz)
    points_2d = project_points(camera, convert_to_tensor(points_3d))
    points_2d = points_2d[0, :, :2]

    c = points_2d[0]
    x = points_2d[1]
    y = points_2d[2]
    z = points_2d[3]
    ax.plot((c[0], x[0]), (c[1], x[1]), '.-', color='red')
    ax.plot((c[0], y[0]), (c[1], y[1]), '.-', color='green')
    ax.plot((c[0], z[0]), (c[1], z[1]), '.-', color='blue')

def draw_probas_edge(image, pred_scores):
    # pred_scores: see render_example documentation
    image_height, image_width, _ = image.shape

    # === convert token positions and probabilities into histogram
    loc_h = pred_scores["loc_h"]/(1024-1)*image_height
    loc_w = pred_scores["loc_w"]/(1024-1)*image_width
    probas_y, _ = np.histogram(loc_h, bins=image_height, range=(0, image_height), weights=pred_scores["scores_h"])
    probas_x, _ = np.histogram(loc_w, bins=image_width, range=(0, image_width), weights=pred_scores["scores_w"])
    eps = 0.1
    probas_y = probas_y / (probas_y.max() + eps) * 255.0 # make colors visible
    probas_x = probas_x / (probas_x.max() + eps) * 255.0 # make colors visible

    # === draw the strip
    stripe_width = 30 # in pixels
    color_y = np.array([1.0, 1.0, 0.0])
    color_strip = probas_y[:, None] * color_y[None, :] # (image_height, 3)
    image[:, 0:stripe_width] = color_strip[:image_height, None, :3] # (height, stripe_width (broadcast), 3)

    color_x = np.array([0.0, 1.0, 0.5])
    color_strip = probas_x[:, None] * color_x[None, :] # (image_width, 3)
    image[0:stripe_width, :] = color_strip # (stripe_width (brooadcast), width, 3)

    return image

def render_example(image, label, prediction: str=None, camera=None, enc=None, enc_pred=None, text: str=None):
    """
    Render an example image to HTML, e.g., for use in notebooks.
    Arguments:
        image: rgb array or (depth, rgb) tuple
        label: the suffix to be predicted
        prediction: the model prediction
        camera: the camera used to record the image
        enc: the default encoder used to decode the both label and prediction
        enc_pred: the encoder that overwrites the default encoder for the prediction
        text: the prefix
    """

    enc_label = enc
    if enc_label is None:
        print("Warning: default action encoder is used")
        enc_label = getActionEncInstance("xyzrotvec-cam-1024xy")
        
    if enc_pred is None:
        enc_pred = enc_label

    if isinstance(image, Image.Image):
        image_width, image_height = image.size
    elif isinstance(image, np.ndarray):
        image_width, image_height, _ = image.shape
    else:
        raise ValueError(f"image was {type(image)}")
        
    if camera is None:
        print("Warning: getting standard camera")
        camera = get_standard_camera(image_width, image_height)

    # logic for when to draw coords
    draw_state_coords = True
    draw_pred_coords = True
    draw_label_coords = False
    if prediction is not None:
        draw_label_coords = False

    plot_width, plot_height = 448, 448

    dpi = 100
    figsize = (plot_width / dpi, plot_height / dpi)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(image)
    ax.axis('off')

    if draw_state_coords:
        robot_state = text.split(" ")[-1]
        draw_coordinate_frame(robot_state, camera, enc_label, ax)        
        
    try:
        curve_25d, quat_c = enc_label.decode_caption(label, camera) 
        curve_2d  = curve_25d[:, :2]
        ax.plot(curve_2d[:, 0], curve_2d[:, 1],'.-', color='green')
        if draw_label_coords:
            draw_coordinate_frame(label, camera, enc_label, ax)            
    except (ValueError, TypeError):
        pass

    html_text = ""
    if text:
       html_text = f'{html.escape("text: "+text)}'
    html_text += f'</br></br>{html.escape("label: "+str(label))}'

    if prediction:
        html_text += f'</br></br>{html.escape("pred: "+prediction)}'
        try:
            curve_2d_gt, quat_c = enc_pred.decode_caption(prediction, camera)
            ax.plot(curve_2d_gt[:, 0], curve_2d_gt[:, 1],'.-', color='lime')
            ax.scatter(curve_2d_gt[0, 0], curve_2d_gt[0, 1], color='red')
            if draw_pred_coords:
                draw_coordinate_frame(prediction, camera, enc_pred, ax)
        except (ValueError, IndexError):
            pass

    with io.BytesIO() as buffer:
        fig.savefig(buffer, format='jpeg',bbox_inches='tight', dpi=dpi)
        image_b64 = str(base64.b64encode(buffer.getvalue()), "utf-8")
        res_str =  f"data:image/jpeg;base64,{image_b64}"
    plt.close(fig)
    return f"""
<div style="display: inline-flex; align-items: center; justify-content: center;">
    <img style="width:224px; height:224px;" src="{res_str}" />
    <p style="width:256px; margin:10px; font-size:small;">{html_text}</p>
</div>
"""
