from skimage.transform import rotate
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from collections import defaultdict
import pydicom as dicom
from enum import Enum
import numpy as np
import json
import os


class DataType(Enum):
    LA2CH = 1
    LA4CH = 2
    LALVOT = 3
    SA = 4
    TRA = 5


# ---- Standard vectors ----

def std_normvector(dtype):
    vect = None
    if dtype == DataType.SA:
        vect = np.array([-0.618, 0.629, 0.415])
    elif dtype == DataType.LA2CH:
        vect = np.array([-0.505, -0.327, -0.010])
    elif dtype == DataType.LA4CH:
        vect = np.array([0.0195, -0.339, 0.613])
    elif dtype == DataType.LALVOT:
        vect = np.array([0.337, 0.021, 0.491])
    else:
        return vect
    return vect / np.sqrt(np.dot(vect, vect))


def std_orientationvector(dtype):
    vect = None
    if dtype == DataType.SA:
        vect = np.array([-0.887, 0.079])
    elif dtype == DataType.LA2CH:
        vect = np.array([-0.937, 0.166])
    elif dtype == DataType.LA4CH:
        vect = np.array([0.632, 0.032])
    elif dtype == DataType.LALVOT:
        vect = np.array([-0.0054, -0.635])
    else:
        return vect
    return vect / np.sqrt(np.dot(vect, vect))


# ---- Accessing the files for each view and patient ----

def view_name(dtype):
    if dtype == DataType.SA:
        return 'sa'
    elif dtype == DataType.LA2CH:
        return 'la2ch'
    elif dtype == DataType.LA4CH:
        return 'la4ch'
    elif dtype == DataType.LALVOT:
        return 'la3ch'
    elif dtype == DataType.TRA:
        return 'tra'
    return None  # none of them


def dicom_paths(source_path):
    def paths_from_type(dcm_paths, pid, view):
        p = os.path.join(source_path, pid, view)
        if view == 'sa':
            p = os.path.join(source_path, pid, view, 'images')
        fs = os.listdir(p)  # files in view folder (e.g.: sa)
        dcm_paths[pid][view] = [os.path.join(p, f) for f in fs] 

    dcm_paths = dict()
    # list the folders (patients)
    patient_ids = os.listdir(source_path)
    # get the path to differnt types
    for pid in patient_ids:
        dcm_paths[pid] = dict()
        # adding file paths in the different views
        paths_from_type(dcm_paths, pid,  view_name(DataType.SA))
        paths_from_type(dcm_paths, pid,  view_name(DataType.LA2CH))
        paths_from_type(dcm_paths, pid,  view_name(DataType.LA4CH))
        paths_from_type(dcm_paths, pid,  view_name(DataType.LALVOT))
        paths_from_type(dcm_paths, pid,  view_name(DataType.TRA))

    return dcm_paths


def sample_paths(source_path):
    # returns one representative path to each view
    dcm_paths = dicom_paths(source_path)
    samples = dict()
    for pid in dcm_paths:
        samples[pid] = dict()
        for view in dcm_paths[pid]:
            ps = dcm_paths[pid][view]
            samples[pid][view] = ps[len(ps)//2]
    return samples


# ---- Calculating the view parameters (norm vector, ...) ----

class ViewPlane:
    """
    ImagePosition/Orientation - x, y, and z coordinates of the upper left hand corner of the image; 
    it is the center of the first voxel transmitted. Image Orientation (0020,0037) specifies the 
    direction cosines of the first row and the first column with respect to the patient.
    """
    def __init__(self,
        cx1, cy1, cz1,
        cx2, cy2, cz2  # orientation of the image (cosines)
    ):
        self.proj_ax1 = np.array([cx1, cy1, cz1], dtype=float)
        self.proj_ax2 = np.array([cx2, cy2, cz2], dtype=float)
    
    @classmethod
    def from_dicom(cls, dcm_path):
        dcm = dicom.dcmread(dcm_path, force=True)
        cx1, cy1, cz1, cx2, cy2, cz2 = dcm.data_element('ImageOrientationPatient')
        return cls(cx1, cy1, cz1, cx2, cy2, cz2)
    
    def normvector(self):
        nv = np.cross(self.proj_ax1, self.proj_ax2)
        return nv / np.sqrt(np.dot(nv, nv))
    
    def xy_plane_per_view(self):
        # defines a new 2D coordinate system for the view plane
        z_vector = np.array([0, 0, 1])  # this is parallel with TRA
        norm_vector = self.normvector()
        x_prime = np.cross(z_vector, norm_vector)
        y_prime = np.cross(norm_vector, x_prime)
        return x_prime / np.sqrt(np.dot(x_prime, x_prime)), y_prime / np.sqrt(np.dot(y_prime, y_prime))
    
    def proj_ax1_2D_coords_in_view(self):
        xp, yp = self.xy_plane_per_view()
        x = np.dot(xp, self.proj_ax1)
        y = np.dot(yp, self.proj_ax1)
        length = np.sqrt(x ** 2 + y ** 2)
        return np.array([x / length, y / length])
    
    def calculate_view(self):
        norm_vector = self.normvector()
        max_view = None
        max_proj = None
        for dtype in DataType:
            std_vct = std_normvector(dtype)
            if std_vct is None:
                continue
            proj = np.abs(np.dot(norm_vector, std_vct))
            if max_proj == None or max_proj < proj:
                max_proj = proj
                max_view = dtype
        return max_view

    def rotation_angle(self):
        is_mirroring_req = False
        # find the view (according to: the closest standard view vector)
        norm_vector = self.normvector()
        max_view = None
        max_proj = None
        for dtype in DataType:
            std_vct = std_normvector(dtype)
            if std_vct is None:
                continue
            proj = np.abs(np.dot(norm_vector, std_vct))
            if max_proj == None or max_proj < proj:
                max_proj = proj
                max_view = dtype
        # decide if mirroring is required
        std_vct = std_normvector(max_view)
        proj = np.dot(norm_vector, std_vct)
        if proj < 0:
            is_mirroring_req = True
        # find the necessary rotation
        std_vct = std_orientationvector(max_view)
        row_vct = self.proj_ax1_2D_coords_in_view()
        cos_rot = np.dot(row_vct, std_vct)
        if np.cross(row_vct, std_vct) > 0:
            rad_rot = np.arccos(cos_rot)
        else:
            rad_rot = -np.arccos(cos_rot)
        return rad_rot, is_mirroring_req, max_view

    def standardize_image(self, image):
        rot_rad, is_mirroring_req, view = self.rotation_angle()
        img_transformed = image.copy()
        if is_mirroring_req:
            rot_rad = -rot_rad
            if view.name == 'LALVOT':
                img_transformed = img_transformed[::-1, :]
            else:
                img_transformed = img_transformed[:, ::-1]
        img_transformed = rotate(img_transformed, rot_rad * 180 / np.pi)
        return img_transformed

    @staticmethod
    def plot_2d_vectors(vectors):
        # vectors: list of numpy 2D vectors
        fig, ax = plt.subplots()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        for vect in vectors:
            ax.quiver(0, 0, *vect[0], color=vect[1], scale=5.0)
        plt.show()

    @staticmethod
    def plot_3d_vectors(vectors):
        # vectors: list of numpy 3D vectors
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        for vect in vectors:
            ax.quiver(0, 0, 0, *vect[0], color=vect[1], length=0.1, normalize=True)
        plt.show()


def main_examine_angles():
    source_path = r'D:\AI\works\Heart\data\hypertrophy\standardization'
    samples = sample_paths(source_path)
    colors = {
        'tra': 'k',
        'sa': 'y',
        'la4ch': 'r',
        'la2ch': 'b',
        'la3ch': 'g'
    }
    vectors_3d = list()
    vectors_2d = list()
    std_3d_vecs = defaultdict(int)
    std_2d_vecs = defaultdict(int)
    for pt in samples:
        for view in samples[pt]:
            v = ViewPlane.from_dicom(samples[pt][view])
            vectors_3d.append([v.normvector(), colors[view]])
            vectors_2d.append([ViewPlane.from_dicom(samples[pt][view]).proj_ax1_2D_coords_in_view(), colors[view]])
            std_3d_vecs[view] += v.normvector() / 20.0
            std_2d_vecs[view] += ViewPlane.from_dicom(samples[pt][view]).proj_ax1_2D_coords_in_view() / 20.0
    vectors_3d_std = list()
    vectors_2d_std = list()
    for view, vect in std_3d_vecs.items():
        vectors_3d_std.append([vect, colors[view]])
    for view, vect in std_2d_vecs.items():
        vectors_2d_std.append([vect, colors[view]])
    ViewPlane.plot_3d_vectors(vectors_3d)
    ViewPlane.plot_3d_vectors(vectors_3d_std)
    ViewPlane.plot_2d_vectors(vectors_2d)
    ViewPlane.plot_2d_vectors(vectors_2d_std)

    with open('std_view_vecs.json', 'wt') as js:
        json.dump({view: vect.tolist() for view, vect in std_3d_vecs.items()}, js)
    
    with open('std_row_vecs.json', 'wt') as js:
        json.dump({view: vect.tolist() for view, vect in std_2d_vecs.items()}, js)


def main_image_rotation_test():
    source_path = r'D:\AI\works\Heart\data\hypertrophy\standardization'
    samples = sample_paths(source_path)
    for pt in samples:
        for view in samples[pt]:
            if view != 'sa':
                continue
            dcm_path = samples[pt][view]
            vp = ViewPlane.from_dicom(dcm_path)
            rot_rad, is_mirroring_req, predicted_view = vp.rotation_angle()
            print(predicted_view.name, view)
            img_orig = dicom.dcmread(dcm_path, force=True).pixel_array
            img_transformed = img_orig.copy()
            if is_mirroring_req:
                rot_rad = -rot_rad
                if predicted_view.name == 'LALVOT':
                    img_transformed = img_transformed[::-1, :]
                else:
                    img_transformed = img_transformed[:, ::-1]
            img_transformed = rotate(img_transformed, rot_rad * 180 / np.pi)
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
            ax1.imshow(img_orig, cmap='gray')
            ax2.imshow(img_transformed, cmap='gray')
            plt.show()


if __name__ == '__main__':
    main_image_rotation_test()
