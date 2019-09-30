import numpy as np
import trimesh
import cv2
import glob
import os, random
from utils.utils_multiobj import threed_correspondences, threed_corners, threed_vertices, \
    compute_projection, pnp
import pickle
import imgaug as ia
import shutil
from imgaug import augmenters as iaa

########## DESCRIPTION
########## Creating dataset where training has multiple objects classes (15) and each train
########## image has one object class and one instance of that. annotations have corner points of the cuboids
########## and generates data for each class in separate folders

num_points_mesh = 9


def generate_data(classes, dataset='LINEMOD'):
    if dataset=='LINEMOD':

        cam_K = np.reshape([572.4114, 0.0, 325.2611, 0.0, 573.57043, 242.04899, 0.0, 0.0, 1.0], (3, 3))
        mesh_base = '../data/LINEMOD_original/models'

        fg_image_dataset = []
        fg_mask_dataset = []
        fg_corners_dataset = []

        for cls in classes:
            mesh_name = os.path.join(mesh_base, cls + '.ply')
            mesh = trimesh.load(mesh_name)
            corners_3d = trimesh.bounds.corners(mesh.bounding_box.bounds)
            corners_3d = np.insert(corners_3d, 0, np.average(corners_3d, axis=0), axis=0)

            trainset_filename = '../data/LINEMOD_original/training_range/' + cls + '.txt'
            train_set = np.loadtxt(trainset_filename, dtype=int, ndmin=1)
            print('Number of real training images as training images in {:s} class is: {:d}'.format(cls, train_set.shape[0]))

            fg_rgbbase = '../data/LINEMOD_original/objects/' + cls + '/rgb'
            fg_maskbase = '../data/LINEMOD_original/objects/' + cls + '/mask'
            fg_posebase = '../data/LINEMOD_original/objects/' + cls + '/pose'

            fg_image_class = []
            fg_mask_class = []
            fg_corners_class = []

            for count, fg_idx in enumerate(train_set):
                fg_idx = '{:04d}'.format(fg_idx)
                fg_imname = os.path.join(fg_rgbbase, fg_idx + '.jpg')
                fg_maskname = os.path.join(fg_maskbase, fg_idx + '.png')
                fg_posename = os.path.join(fg_posebase, fg_idx + '.txt')

                pose_mat = np.loadtxt(fg_posename, dtype=float, ndmin=2)
                corners_2d_np = corners_transform(corners_3d, cam_K, pose_mat[0:3, :])

                fg_image_class.append(cv2.imread(fg_imname))
                fg_mask_class.append(cv2.imread(fg_maskname) / 255)
                fg_corners_class.append(corners_2d_np)

            fg_image_dataset.append(fg_image_class)
            fg_mask_dataset.append(fg_mask_class)
            fg_corners_dataset.append(fg_corners_class)

        return fg_image_dataset, fg_mask_dataset, fg_corners_dataset
    else:
        return None


def corners_transform(corners, cam_k, cam_rt):
    cam_p = np.matmul(cam_k, cam_rt)
    corners_homo = np.concatenate((corners, np.ones((num_points_mesh, 1))), axis=1)
    corners_transformed_homo = np.matmul(cam_p, corners_homo.T)
    corners_transformed = np.zeros((2, num_points_mesh))
    corners_transformed[0, :] = corners_transformed_homo[0, :]/corners_transformed_homo[2, :]
    corners_transformed[1, :] = corners_transformed_homo[1, :]/corners_transformed_homo[2, :]

    return corners_transformed


def create_annotations(aug_annotbase, file_id, corners, class_name):
    annotation = []

    for num_obj in range(corners.shape[0]):
        gt_instance = list(np.reshape(corners[num_obj], (num_points_mesh*2), 'F'))
        gt_instance.append(class_name[num_obj])
        annotation.append(gt_instance)

    annotation = np.array(annotation, dtype=str, ndmin=2)
    aug_annotname = os.path.join(aug_annotbase, file_id+'.txt')
    np.savetxt(aug_annotname, annotation, fmt='%s')


def create_annotations_test(aug_annotbase, file_id, corners, class_name):
    gt_instance = list(np.reshape(corners, (num_points_mesh * 2), 'F'))
    gt_instance.append(class_name)
    gt_instance = np.array(gt_instance, dtype=str, ndmin=2)
    aug_annotname = os.path.join(aug_annotbase, file_id+'.txt')
    np.savetxt(aug_annotname, gt_instance, fmt='%s')


if __name__ == "__main__":
    class_name = ('ape', 'benchvise', 'cam', 'can', 'cat', 'driller', 'duck', 'eggbox', 'glue',
                  'holepuncher', 'iron', 'lamp', 'phone')

    # # # #### Training data generate
    fg_image_dataset, fg_mask_dataset, fg_corners_dataset = generate_data(class_name, 'LINEMOD')
    linemod_cls_names={'ape':0,'cam':1,'cat':2,'duck':3,'glue':4,'iron':5,'phone':6, 'benchvise':7,'can':8,'driller':9,'eggbox':10,'holepuncher':11,'lamp':12}

    mesh_base = '../data/LINEMOD_original/models'

    for cls_idx, cls in enumerate(class_name):
        count = 0
        trainval = []

        aug_imagebase = '../data/LINEMOD_singleobj_realsyn/' + cls + '/JPEGImages'
        aug_annotbase = '../data/LINEMOD_singleobj_realsyn/' + cls + '/Annotations'
        pose_annotbase = '../data/LINEMOD_singleobj_realsyn/' + cls + '/PoseAnnotations'
        mask_annotbase = '../data/LINEMOD_singleobj_realsyn/' + cls + '/Masks'

        if not os.path.exists(aug_imagebase):
            os.makedirs(aug_imagebase)
        if not os.path.exists(aug_annotbase):
            os.makedirs(aug_annotbase)
        if not os.path.exists(pose_annotbase):
            os.makedirs(pose_annotbase)
        if not os.path.exists(mask_annotbase):
            os.makedirs(mask_annotbase)
        if not os.path.exists('../data/LINEMOD_singleobj_realsyn/' + cls + '/ImageSets'):
            os.makedirs('../data/LINEMOD_singleobj_realsyn/' + cls + '/ImageSets')

        corners_3d = threed_corners(cls)

        if num_points_mesh == 9:
            objpoints3D = corners_3d
        else:
            objpoints3D = threed_correspondences(cls, '../data/LINEMOD_singleobj_realsyn_surface')

        mesh_name = os.path.join(mesh_base, cls + '.ply')
        mesh = trimesh.load(mesh_name)

        mesh = mesh.subdivide()
        vertices = mesh.vertices
        vertices = np.concatenate((np.transpose(vertices), np.ones((1, vertices.shape[0]))), axis=0)

        cam_K = np.reshape([572.4114, 0.0, 325.2611, 0.0, 573.57043, 242.04899, 0.0, 0.0, 1.0], (3, 3))

        for cls_instance in range(len(fg_image_dataset[cls_idx])):

            class_names = []
            fg_corners = []

            fg_image = fg_image_dataset[cls_idx][cls_instance]

            fg_mask = fg_mask_dataset[cls_idx][cls_instance]
            fg_corners.append(np.array(fg_corners_dataset[cls_idx][cls_instance]))

            class_names.append(cls)

            aug_image = np.array(fg_image, dtype='uint8')
            fg_corners = np.array(fg_corners, dtype='float')

            R, t = pnp(corners_3d, fg_corners[0].T, cam_K)
            pose_mat = np.concatenate((R, t), axis=1)
            corners_2d = corners_transform(corners_3d, cam_K, pose_mat)

            pose = list(pose_mat[:3, :].reshape((12)))
            pose.append(cls)

            annot_name = os.path.join(mask_annotbase,  str(count) + 'linemod_ori.jpg')
            cv2.imwrite(annot_name+'.jpg', fg_mask * 255)

            annot_name = os.path.join(pose_annotbase,  str(count) + 'linemod_ori.txt')
            aug_filename = os.path.join(aug_imagebase, str(count) + 'linemod_ori.jpg')
            cv2.imwrite(aug_filename, aug_image)
            create_annotations(aug_annotbase, str(count) + 'linemod_ori', fg_corners, class_names)
            np.savetxt(annot_name, np.array(pose, dtype=str, ndmin=2), fmt='%s')

            trainval.append(str(count) + 'linemod_ori')
            count += 1

        trainval = np.array(trainval, dtype='str')
        np.savetxt('../data/LINEMOD_singleobj_realsyn/' + cls + '/ImageSets/trainerror.txt', trainval, delimiter=" ",
                   fmt="%s")