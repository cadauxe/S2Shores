# -*- coding: utf-8 -*-
""" Unit tests of the directional_array module

:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file.
:created: Jun 1, 2021
"""
import unittest
import numpy as np  # @NoMove

from bathyinversionvagues.generic_utils.directional_array import DirectionalArray

TEST_ARRAY1 = np.array([[1, 9, 4],
                        [4., 7, -8],
                        [3., -35, 0],
                        [5.3, -1.7, 0.01]
                        ])


class UTestDirectionalArray(unittest.TestCase):
    """ Test class for DirectionalArray class """

    def test_n_constructor(self) -> None:
        """ Test the constructor of DirectionalArray
        """
        # Array and directions specified.
        test_array = DirectionalArray()
        test_array.insert_from_arrays(TEST_ARRAY1, np.array([4, -11, 100.]))
        self.assertEqual(test_array.nb_directions, 3)
        array_out, directions = test_array.get_as_arrays()
        self.assertEqual(len(directions), 3)
        self.assertEqual(array_out.shape, (4, 3))
        self.assertEqual(array_out.dtype, np.float64)

    def test_d_constructor(self) -> None:
        """ Test the constructor of DirectionalArray in degraded cases
        """
        # Array specified but of wrong types
        test_array = DirectionalArray()
        with self.assertRaises(TypeError) as excp:
            test_array.insert_from_arrays(np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4]))
        expected = 'array for a DirectionalArray must be a 2D numpy array'
        self.assertEqual(str(excp.exception), expected)

        # Provided directions have not the same number of elements than the array column number
        with self.assertRaises(ValueError) as excp:
            test_array.insert_from_arrays(np.empty((10, 6)), np.array([1, 2, 3, 4]))
        expected = 'directions size must be equal to the number of columns of the array'
        self.assertEqual(str(excp.exception), expected)

        # Provided directions have values which are too close with respect to the direction_step
        with self.assertRaises(ValueError) as excp:
            test_array.insert_from_arrays(np.empty((10, 4)), np.array([1, 2.50001, 3.4, 4]))
        expected = 'dimensions after quantization has not the same number of elements (3)'
        expected += ' than the number of columns in the array (4)'
        self.assertEqual(str(excp.exception), expected)

    def test_n_indexing(self) -> None:
        """ Test the [] operator of DirectionalArray
        """
        # Array for this test.
        test_array = DirectionalArray()
        test_array.insert_from_arrays(TEST_ARRAY1, np.array([4, -11, 100.]))

        # Retrieve an existing direction
        vector = test_array[-11]
        self.assertEqual(vector.shape, (4,))
        self.assertEqual(vector.dtype, np.float64)
        self.assertTrue(np.array_equal(vector, np.array([9., 7., -35, -1.7])))
