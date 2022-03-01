from unittest import TestCase

import numpy as np

from src.UtilsCV import estimate_intersection_between_lines, normalize_vectors, estimate_point_of_interest_in_scene, \
    get_sphere_matrix, get_rotation_quaternion_from_vec1_to_vec2, rotate_vec_with_quaternion, \
    get_camera_dir_from_c2w, get_rotation_matrix_from_v1_to_v2, get_rays_directions


class Test(TestCase):
    """
    Some tests for general mathematical functions that are used in UtilsCV.
    """

    def test_normalize_vectors1(self):
        vec = np.asarray([1, 1])
        real_normalized = np.asarray([0.7071, 0.7071])
        normalized = normalize_vectors(vec)
        self.assertTrue(np.allclose(normalized, real_normalized))

    def test_normalize_vectors2(self):
        vecs = np.asarray([
            [1, 1],
            [1, 0],
            [0, 1],
        ])
        real_normalized = np.asarray([
            [0.7071, 0.7071],
            [1, 0],
            [0, 1],
        ])
        normalized = normalize_vectors(vecs)
        self.assertTrue(np.allclose(normalized, real_normalized))

    def test_estimate_3d_intersection(self):
        dirs = np.asarray([
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 0],
            [0, 1],
        ])
        location_on_points = np.asarray([
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 1],
            [1, 0],
        ])
        dirs_and_t = np.stack([dirs, location_on_points], axis=1)  # Stack to list of pairs of directions and points

        estimated_intersection = estimate_intersection_between_lines(dirs_and_t)
        real_intersection = np.array([1, 1])
        self.assertTrue(np.allclose(real_intersection, estimated_intersection))

    def test_estimate_point_of_interest_in_scene1(self):
        c2w1 = get_sphere_matrix(1, 0, 0, 0)
        c2w2 = get_sphere_matrix(1, 0, 90, 0)
        estimated_intersection, is_there_intersection = estimate_point_of_interest_in_scene([c2w1, c2w2])

        self.assertTrue(is_there_intersection)
        self.assertTrue(np.allclose(estimated_intersection, np.zeros_like(estimated_intersection)))

    def test_estimate_point_of_interest_in_scene2(self):
        c2w1 = get_sphere_matrix(1, 0, 0, 0)
        c2w2 = get_sphere_matrix(1, 90, 0, 0)
        c2w3 = get_sphere_matrix(1, 0, 90, 0)
        c2w4 = get_sphere_matrix(1, 0, 0, 90)
        c2w5 = get_sphere_matrix(1, 0, 0, 90)

        # Create and translate the matrices:
        c2w_outlier1 = get_sphere_matrix(1, 0, 90, 0)
        c2w_outlier1[:3, 3] += 1

        c2w_outlier2 = get_sphere_matrix(1, 0, 90, 0)
        c2w_outlier2[:3, 3] -= 1

        estimated_intersection, is_there_intersection = estimate_point_of_interest_in_scene([
            c2w1,
            c2w2,
            c2w3,
            c2w4,
            c2w5,
            c2w_outlier1,
            c2w_outlier2
        ])

        self.assertTrue(is_there_intersection)
        self.assertTrue(np.allclose(estimated_intersection, np.zeros_like(estimated_intersection)))

    def test_estimate_point_of_interest_in_scene3(self):
        # Create and translate the matrices:
        c2w1 = get_sphere_matrix(1, 0, 0, 0)
        c2w1[:3, 3] += 1
        c2w2 = get_sphere_matrix(1, 90, 0, 0)
        c2w2[:3, 3] += 1
        c2w3 = get_sphere_matrix(1, 0, 90, 0)
        c2w3[:3, 3] += 1
        c2w4 = get_sphere_matrix(1, 0, 0, 90)
        c2w4[:3, 3] += 1
        c2w5 = get_sphere_matrix(1, 0, 0, 90)
        c2w5[:3, 3] += 1

        c2w_outlier1 = get_sphere_matrix(1, 0, 90, 0)
        c2w_outlier2 = get_sphere_matrix(1, 0, 90, 0)

        estimated_intersection, is_there_intersection = estimate_point_of_interest_in_scene([
            c2w1,
            c2w2,
            c2w3,
            c2w4,
            c2w5,
            c2w_outlier1,
            c2w_outlier2
        ])

        self.assertTrue(is_there_intersection)
        self.assertTrue(np.allclose(estimated_intersection, np.ones_like(estimated_intersection)))

    def test_estimate_point_of_interest_in_scene4(self):
        c2w1 = get_sphere_matrix(1, 0, 0, 0)
        c2w1[:3, 3] += 1
        c2w2 = get_sphere_matrix(1, 90, 0, 0)
        c2w2[:3, 3] -= 1
        c2w3 = get_sphere_matrix(1, 0, 90, 0)

        estimated_intersection, is_there_intersection = estimate_point_of_interest_in_scene([
            c2w1,
            c2w2,
            c2w3
        ])

        self.assertFalse(is_there_intersection)

    def test_get_rotation_quaternion_from_vec1_to_vec2(self):
        v1 = np.asarray([1, 0, 0])
        v2 = np.asarray([0, 1/np.sqrt(2), 1/np.sqrt(2)])
        # The angle between v1 and v2 is pi/2.
        result_q = get_rotation_quaternion_from_vec1_to_vec2(v1, v2)
        expected = np.array([1/np.sqrt(2), 0, -0.5, 0.5])
        self.assertTrue(np.allclose(result_q, expected))
        self.assertTrue(np.allclose(v2, rotate_vec_with_quaternion(v1, result_q)))

    def test_get_camera_dir_from_c2w(self):
        c2w = get_sphere_matrix(1, 90, 0, 0)
        direction = np.asarray([0, 1, 0])   # c2w looking up in the y-axis direction.
        self.assertTrue(np.allclose(direction, get_camera_dir_from_c2w(c2w)))

    def test_get_rotation_matrix_from_v1_to_v2_test1(self):
        m1, m2 = get_sphere_matrix(1, 90, 0, 0), get_sphere_matrix(1, 0, 0, 0)
        v1 = get_camera_dir_from_c2w(m1)
        v2 = get_camera_dir_from_c2w(m2)
        rotation = get_rotation_matrix_from_v1_to_v2(v1, v2)
        self.assertTrue(np.allclose(v2, rotation @ v1))

        m3 = get_sphere_matrix(1, 0, 90, 0)
        v3 = get_camera_dir_from_c2w(m3)

        # Because v3 is perpendicular to v1 and v2, the rotation should not affect v3.
        self.assertTrue(np.allclose(v3, rotation @ v3))

    def test_get_rotation_matrix_from_v1_to_v2_test2(self):
        v1 = np.asarray([1, 0, 0])
        v2 = np.asarray([0, 1 / np.sqrt(2), 1 / np.sqrt(2)])
        # The angle between v1 and v2 is pi/2.
        rotation = get_rotation_matrix_from_v1_to_v2(v1, v2)
        self.assertTrue(np.allclose(v2, rotation @ v1))

    def test_get_rotation_matrix_from_v1_to_v2_test3(self):
        m1, m2 = get_sphere_matrix(1, 45, 0, 0), get_sphere_matrix(1, 0, 0, 0)
        v1 = get_camera_dir_from_c2w(m1)
        v2 = get_camera_dir_from_c2w(m2)
        rotation = get_rotation_matrix_from_v1_to_v2(v1, v2)
        self.assertTrue(np.allclose(v2, rotation @ v1))

        result_q = get_rotation_quaternion_from_vec1_to_vec2(v1, v2)
        self.assertTrue(np.allclose(v2, rotate_vec_with_quaternion(v1, result_q)))

    def test_get_rotation_matrix_from_v1_to_v2_test4(self):
        m1, m2 = get_sphere_matrix(1, 33, 133, 33), get_sphere_matrix(1, 5, 243, 12)
        v1 = get_camera_dir_from_c2w(m1)
        v2 = get_camera_dir_from_c2w(m2)
        rotation = get_rotation_matrix_from_v1_to_v2(v1, v2)
        self.assertTrue(np.allclose(v2, rotation @ v1))

        result_q = get_rotation_quaternion_from_vec1_to_vec2(v1, v2)
        self.assertTrue(np.allclose(v2, rotate_vec_with_quaternion(v1, result_q)))