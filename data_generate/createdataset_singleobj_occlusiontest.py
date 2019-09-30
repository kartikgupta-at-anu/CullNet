import numpy as np
import cv2, glob, os, math, trimesh
import yaml
import utils.hinter_flip as hinter_flip
import utils.transform as transform
from utils.utils_multiobj import threed_corners, corners_transform, convert_to_occlusionfmt, load_gt_pose_brachmann

########## DESCRIPTION
########## Creating dataset for occlusion testing but only for single class where in occlusion dataset only one class is annotated

num_points_mesh = 9


def create_annotations(aug_annotbase, file_id, corners, class_name):
    annotation = []

    for num_obj in xrange(corners.shape[0]):
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
    ### Testing data generate
    cam_K = np.reshape([572.4114, 0.0, 325.2611, 0.0, 573.57043, 242.04899, 0.0, 0.0, 1.0], (3, 3))
    classes = ('Ape', 'Can', 'Cat', 'Driller', 'Duck', 'Eggbox', 'Glue', 'Holepuncher')
    mesh_base = '../data/OcclusionChallengeICCV2015/models'

    for cls_id, cls in enumerate(classes):
        aug_annotbase = '../data/singleobject_occtest/' + cls.lower() + '/Annotations'
        aug_imagebase = '../data/singleobject_occtest/' + cls.lower() + '/JPEGImages'
        pose_base = '../data/OcclusionChallengeICCV2015/poses/' + cls
        test = []

        if not os.path.exists(aug_imagebase):
            os.makedirs(aug_imagebase)
        if not os.path.exists(aug_annotbase):
            os.makedirs(aug_annotbase)
        if not os.path.exists('../data/singleobject_occtest/' + cls.lower() + '/ImageSets'):
            os.makedirs('../data/singleobject_occtest/' + cls.lower() + '/ImageSets')

        corners_3d = threed_corners(cls.lower())
        corners_cuboid = corners_3d
        corners_3d, corners_cuboid = convert_to_occlusionfmt(corners_3d, corners_cuboid)
        R_model = transform.rotation_matrix(math.pi, [0, 1, 0])[:3, :3]
        # # Extra rotation around Z axis by pi for some models
        R_z = transform.rotation_matrix(math.pi, [0, 0, 1])[:3, :3]
        R_model = R_z.dot(R_model)

        # The ground truth poses of Brachmann et al. are related to a different
        # model coordinate system - to get the original Hinterstoisser's orientation
        # of the objects, we need to rotate by pi/2 around X and by pi/2 around Z
        R_z_90 = transform.rotation_matrix(-math.pi * 0.5, [0, 0, 1])[:3, :3]
        R_x_90 = transform.rotation_matrix(-math.pi * 0.5, [1, 0, 0])[:3, :3]
        R_conv = np.linalg.inv(R_model.dot(R_z_90.dot(R_x_90)))

        for imname in glob.iglob('../data/OcclusionChallengeICCV2015/RGB-D/rgb_noseg/*.png'):
            idx = os.path.basename(imname)[6:-4]
            image = cv2.imread(imname)
            corners_2d = {}

            pose_filename = os.path.join(pose_base, 'info_' + idx + '.txt')

            pose = load_gt_pose_brachmann(pose_filename)
            if pose['R'].size != 0 and pose['t'].size != 0:
                R_m2c = pose['R'].dot(R_conv)
                t_m2c = pose['t']
                pose_mat = np.concatenate((R_m2c, t_m2c), axis=1)
                corners_2d = corners_transform(corners_3d, cam_K, pose_mat)
            else:
                continue;
            aug_filename = os.path.join(aug_imagebase, idx + '_test.jpg')
            cv2.imwrite(aug_filename, image)
            create_annotations_test(aug_annotbase, idx + '_test', corners_2d, cls.lower())
            test.append(idx+'_test')

        test = np.array(test, dtype='string')
        np.savetxt('../data/singleobject_occtest/'+ cls.lower() +'/ImageSets/test.txt', test, delimiter=" ", fmt="%s")
