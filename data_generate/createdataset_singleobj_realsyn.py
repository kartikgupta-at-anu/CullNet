import numpy as np
import trimesh
import cv2
import glob
import os, random
from utils.utils_multiobj import threed_correspondences, threed_corners, threed_vertices, compute_projection, pnp
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
                corners_2d = [ia.Keypoint(x=point[0], y=point[1]) for point in corners_2d_np.transpose()]
                corners_2d = ia.KeypointsOnImage(corners_2d, shape=cv2.imread(fg_imname).shape)

                fg_image_class.append(cv2.imread(fg_imname))
                fg_mask_class.append(cv2.imread(fg_maskname) / 255)
                fg_corners_class.append(corners_2d)

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
    bg_database_list = glob.glob('../data/VOCdevkit/VOC2007-12/JPEGImages/*.jpg')

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

        cam_K = np.array([[700.,    0.,  320.],
                          [0.,  700.,  240.],
                          [0.,    0.,    1.]])
        database_list = glob.iglob(('../data/LINEMOD/renders/{:s}/*.jpg').format(cls))

        for cls_instance in database_list:
            class_names = []
            fg_corners = []

            imname = cls_instance
            annotname = cls_instance[:-4] + '_RT.pkl'

            pose_mat = pickle.load(open(annotname, 'rb'))['RT']
            corners_2d = corners_transform(corners_3d, cam_K, pose_mat)

            fg_corners.append(corners_2d)
            class_names.append(cls)

            fg_corners = np.array(fg_corners, dtype='float')
            pose = list(pose_mat[:3, :].reshape((12)))
            pose.append(cls)

            mask_image_appended = np.zeros([1000, 1000, 3]).astype('uint8')
            vertices2d = compute_projection(vertices, pose_mat[0:3, :], cam_K)
            vertices2d = vertices2d.astype('int')
            mask_image_appended[vertices2d[1] + 200, vertices2d[0] + 200] = 255, 255, 255
            mask_image = mask_image_appended[200:680, 200:840]

            mask_name = os.path.join(mask_annotbase,  str(count) + '.jpg')
            cv2.imwrite(mask_name+'.jpg', mask_image * 255)

            annot_name = os.path.join(pose_annotbase,  str(count) + '.txt')
            aug_filename = os.path.join(aug_imagebase, str(count) + '.jpg')
            shutil.copyfile(imname, aug_filename)
            create_annotations(aug_annotbase, str(count), fg_corners, class_names)
            np.savetxt(annot_name, np.array(pose, dtype=str, ndmin=2), fmt='%s')

            trainval.append(str(count))
            count += 1

        cam_K = np.reshape([572.4114, 0.0, 325.2611, 0.0, 573.57043, 242.04899, 0.0, 0.0, 1.0], (3, 3))

        for cls_instance in range(len(fg_image_dataset[cls_idx])):
            bg_imnames = random.sample(bg_database_list, 50)

            fg_images_aug = [fg_image_dataset[cls_idx][cls_instance]] * 50
            fg_masks_aug = [fg_mask_dataset[cls_idx][cls_instance]] * 50
            fg_corners_aug = [fg_corners_dataset[cls_idx][cls_instance]] * 50

            sometimes = lambda aug: iaa.Sometimes(0.6, aug)
            seq = iaa.Sequential(
                [
                    sometimes(iaa.Affine(
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                        rotate=(-15, 15) # rotate by -90 to +90 degrees
                    ))])
            seq_det = seq.to_deterministic() # call this for each batch again, NOT only once at the start

            fg_images_aug = seq_det.augment_images(fg_images_aug)
            fg_masks_aug = seq_det.augment_images(fg_masks_aug)
            fg_corners_aug = seq_det.augment_keypoints(fg_corners_aug)

            for aug_num in range(50):
                obj_center = []
                class_names = []
                fg_corners = []

                fg_image = fg_images_aug[aug_num]
                bg_image = cv2.imread(bg_imnames[aug_num])
                bg_image = cv2.resize(bg_image, (fg_image.shape[1], fg_image.shape[0]))

                fg_image = fg_images_aug[aug_num]
                fg_mask = fg_masks_aug[aug_num]
                fg_corners.append(np.array([[point.x, point.y] for point in
                                            fg_corners_aug[aug_num].keypoints]).transpose())

                bg_mask = 1. - fg_mask
                bg_image = np.multiply(fg_mask, fg_image) + np.multiply(bg_mask, bg_image)
                class_names.append(cls)

                aug_image = np.array(bg_image, dtype='uint8')
                fg_corners = np.array(fg_corners, dtype='float')

                if (any(val > aug_image.shape[1] for val in fg_corners[:, 0, :].flatten()) or
                        any(val < 0 for val in fg_corners[:, 0, :].flatten()) or
                        any(val > aug_image.shape[0] for val in fg_corners[:, 1, :].flatten()) or
                        any(val < 0 for val in fg_corners[:, 1, :].flatten())):
                    continue

                R, t = pnp(corners_3d, fg_corners[0].T, cam_K)
                pose_mat = np.concatenate((R, t), axis=1)
                corners_2d = corners_transform(corners_3d, cam_K, pose_mat)

                pose = list(pose_mat[:3, :].reshape((12)))
                pose.append(cls)

                annot_name = os.path.join(mask_annotbase,  str(count) + '.jpg')
                cv2.imwrite(annot_name+'.jpg', fg_mask * 255)

                annot_name = os.path.join(pose_annotbase,  str(count) + '.txt')
                aug_filename = os.path.join(aug_imagebase, str(count) + '.jpg')
                cv2.imwrite(aug_filename, aug_image)
                create_annotations(aug_annotbase, str(count), fg_corners, class_names)
                np.savetxt(annot_name, np.array(pose, dtype=str, ndmin=2), fmt='%s')

                trainval.append(str(count))
                count += 1

        trainval = np.array(trainval, dtype='str')
        np.savetxt('../data/LINEMOD_singleobj_realsyn/' + cls + '/ImageSets/trainval.txt', trainval, delimiter=" ",
                   fmt="%s")

    cam_K = np.reshape([572.4114, 0.0, 325.2611, 0.0, 573.57043, 242.04899, 0.0, 0.0, 1.0], (3, 3))

    #### Validation data generate

    for cls_idx, cls in enumerate(class_name):
        count = 0
        val = []

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

        for cls_instance in range(len(fg_image_dataset[cls_idx])):
            bg_imnames = random.sample(bg_database_list, 5)

            fg_images_aug = [fg_image_dataset[cls_idx][cls_instance]] * 8
            fg_masks_aug = [fg_mask_dataset[cls_idx][cls_instance]] * 8
            fg_corners_aug = [fg_corners_dataset[cls_idx][cls_instance]] * 8

            sometimes = lambda aug: iaa.Sometimes(1, aug)
            seq = iaa.Sequential(
                [
                    sometimes(iaa.Affine(
                        # scale=(0.9, 1.1), # scale images to 90-110% of their size, individually per axis
                        translate_percent={"x": (-0.3, 0.3), "y": (-0.3, 0.3)}, # translate by -20 to +20 percent (per axis)
                        rotate=(-30, 30) # rotate by -90 to +90 degrees
                    ))])
            seq_det = seq.to_deterministic() # call this for each batch again, NOT only once at the start

            fg_images_aug = seq_det.augment_images(fg_images_aug)
            fg_masks_aug = seq_det.augment_images(fg_masks_aug)
            fg_corners_aug = seq_det.augment_keypoints(fg_corners_aug)

            for aug_num in range(5):
                obj_center = []
                class_names = []
                fg_corners = []

                fg_image = fg_images_aug[aug_num]
                bg_image = cv2.imread(bg_imnames[aug_num])
                bg_image = cv2.resize(bg_image, (fg_image.shape[1], fg_image.shape[0]))

                fg_image = fg_images_aug[aug_num]
                fg_mask = fg_masks_aug[aug_num]
                fg_corners.append(np.array([[point.x, point.y] for point in
                                            fg_corners_aug[aug_num].keypoints]).transpose())

                bg_mask = 1. - fg_mask
                bg_image = np.multiply(fg_mask, fg_image) + np.multiply(bg_mask, bg_image)
                class_names.append(cls)

                aug_image = np.array(bg_image, dtype='uint8')
                fg_corners = np.array(fg_corners, dtype='float')

                if (any(val > aug_image.shape[1] for val in fg_corners[:, 0, :].flatten()) or
                        any(val < 0 for val in fg_corners[:, 0, :].flatten()) or
                        any(val > aug_image.shape[0] for val in fg_corners[:, 1, :].flatten()) or
                        any(val < 0 for val in fg_corners[:, 1, :].flatten())):
                    continue

                R, t = pnp(corners_3d, fg_corners[0].T, cam_K)
                pose_mat = np.concatenate((R, t), axis=1)
                pose = list(pose_mat[:3, :].reshape((12)))
                pose.append(cls)
                corners_2d = corners_transform(corners_3d, cam_K, pose_mat)

                annot_name = os.path.join(mask_annotbase,  str(count) + '_val.jpg')
                cv2.imwrite(annot_name+'.jpg', fg_mask * 255)

                annot_name = os.path.join(pose_annotbase,  str(count) + '_val.txt')
                aug_filename = os.path.join(aug_imagebase, str(count) + '_val.jpg')
                cv2.imwrite(aug_filename, aug_image)
                create_annotations(aug_annotbase, str(count) + '_val', fg_corners, class_names)
                np.savetxt(annot_name, np.array(pose, dtype=str, ndmin=2), fmt='%s')

                val.append(str(count) + '_val')
                count += 1

        val = np.array(val, dtype='str')
        np.savetxt('../data/LINEMOD_singleobj_realsyn/' + cls + '/ImageSets/val.txt', val, delimiter=" ",
                   fmt="%s")

    #### Testing data generate

    cam_K = np.reshape([572.4114, 0.0, 325.2611, 0.0, 573.57043, 242.04899, 0.0, 0.0, 1.0], (3, 3))
    mesh_base = '../data/LINEMOD_original/models'

    for cls in class_name:
        test = []

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

        mesh_name = os.path.join(mesh_base, cls + '.ply')
        mesh = trimesh.load(mesh_name)

        corners_3d = trimesh.bounds.corners(mesh.bounding_box.bounds)
        corners_3d = np.insert(corners_3d, 0, np.average(corners_3d, axis=0), axis=0)

        mesh = mesh.subdivide()
        vertices = mesh.vertices
        vertices = np.concatenate((np.transpose(vertices), np.ones((1, vertices.shape[0]))), axis=0)

        trainset_filename = '../data/LINEMOD_original/training_range/' + cls + '.txt'
        train_set = ['{:04d}'.format(idx) for idx in np.loadtxt(trainset_filename, dtype=int, ndmin=1)]
        train_set = set(train_set)
        whole_set = set([os.path.splitext(ind)[0]
                         for ind in os.listdir('../data/LINEMOD_original/objects/' + cls + '/rgb/')])
        test_set = np.array(list(whole_set-train_set), dtype=int, ndmin=1)

        print('Number of total real images in {:s} class is: ' \
              '{:d}'.format(cls, len(whole_set)))

        print('Number of real training images as training images in {:s} class is: ' \
              '{:d}'.format(cls, len(train_set)))

        print('Number of real testing images as testing images in {:s} class is: ' \
              '{:d}'.format(cls, test_set.shape[0]))

        rgb_base = '../data/LINEMOD_original/objects/' + cls + '/rgb'
        pose_base = '../data/LINEMOD_original/objects/' + cls + '/pose'

        for idx in test_set:

            idx = '{:04d}'.format(idx)

            imname = os.path.join(rgb_base, idx + '.jpg')
            posename = os.path.join(pose_base, idx + '.txt')

            image = cv2.imread(imname)
            pose_mat = np.loadtxt(posename, dtype=float, ndmin=2)
            pose = list(pose_mat[:3, :].reshape((12)))
            pose.append(cls)

            corners_2d = corners_transform(corners_3d, cam_K, pose_mat[0:3, :])

            mask_image_appended = np.zeros([1000, 1000, 3]).astype('uint8')
            vertices2d = compute_projection(vertices, pose_mat[0:3, :], cam_K)
            vertices2d = vertices2d.astype('int')
            mask_image_appended[vertices2d[1] + 200, vertices2d[0] + 200] = 255, 255, 255
            mask_image = mask_image_appended[200:680, 200:840]
            annot_name = os.path.join(mask_annotbase,  idx + '_test_' + cls + '.jpg')
            cv2.imwrite(annot_name+'.jpg', mask_image)

            aug_filename = os.path.join(aug_imagebase, idx + '_test_' + cls + '.jpg')
            annot_name = os.path.join(pose_annotbase,  idx + '_test_' + cls + '.txt')
            cv2.imwrite(aug_filename, image)
            create_annotations_test(aug_annotbase, idx + '_test_' + cls, corners_2d, cls)
            np.savetxt(annot_name, np.array(pose, dtype=str, ndmin=2), fmt='%s')

            test.append(idx + '_test_' + cls)

        test = np.array(test, dtype='str')
        np.savetxt('../data/LINEMOD_singleobj_realsyn/' + cls + '/ImageSets/test.txt', test,
                   delimiter=" ", fmt="%s")