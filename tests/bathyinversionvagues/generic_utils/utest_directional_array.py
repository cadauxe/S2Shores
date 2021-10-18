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
        test_array = DirectionalArray(TEST_ARRAY1, np.array([4, -11, 100.]))
        self.assertEqual(test_array.nb_directions, 3)
        self.assertEqual(test_array.array.shape, (4, 3))
        self.assertEqual(test_array.array.dtype, np.float64)

    def test_d_constructor(self) -> None:
        """ Test the constructor of DirectionalArray in degraded cases
        """
        # Array specified but of wrong types
        with self.assertRaises(TypeError) as excp:
            _ = DirectionalArray(np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4]))
        expected = 'array for a DirectionalArray must be a 2D numpy array'
        self.assertEqual(str(excp.exception), expected)

        # Provided directions have not the same number of elements than the array column number
        with self.assertRaises(ValueError) as excp:
            _ = DirectionalArray(np.empty((10, 6)), np.array([1, 2, 3, 4]))
        expected = 'directions size must be equal to the number of columns of the array'
        self.assertEqual(str(excp.exception), expected)

        # Provided directions have values which are too close with respect to the direction_step
        with self.assertRaises(ValueError) as excp:
            _ = DirectionalArray(np.empty((10, 4)), np.array([1, 2.50001, 3.4, 4]))
        expected = 'some dimensions values are too close to each other considering '
        expected += 'the dimensions quantization step: 1.0°'
        self.assertEqual(str(excp.exception), expected)

    def test_n_create_empty(self) -> None:
        """ Test the DirectionalArray.create_empty() class method in nominal cases
        """
        # No directions specified. An empty array with 180 directions is created
        test_array = DirectionalArray.create_empty(10)
        self.assertEqual(test_array.nb_directions, 180)
        self.assertEqual(test_array.array.shape, (10, 180))
        self.assertEqual(test_array.array.dtype, np.float64)

        # No directions specified. An empty array with the number of directions is created
        test_array = DirectionalArray.create_empty(10, directions=np.array([-11, 4, 5., 100.]))
        self.assertEqual(test_array.nb_directions, 4)
        self.assertEqual(test_array.array.shape, (10, 4))
        self.assertEqual(test_array.array.dtype, np.float64)

        # No directions specified. An empty array with the number of directions is created
        # FIXME: Unordered directions are accepted, but not reordered
        test_array = DirectionalArray.create_empty(10, directions=np.array([4, -11, 5., 100.]))
        self.assertEqual(test_array.nb_directions, 4)
        self.assertEqual(test_array.array.shape, (10, 4))
        self.assertEqual(test_array.array.dtype, np.float64)

    def test_d_create_empty(self) -> None:
        """ Test the DirectionalArray.create_empty() class method in degraded cases
        """

        # Directions specified but of wrong types
        with self.assertRaises(AttributeError) as excp:
            _ = DirectionalArray.create_empty(10, directions=[1, 2, 3, 4])
        expected = "'list' object has no attribute 'ndim'"
        self.assertEqual(str(excp.exception), expected)

        with self.assertRaises(TypeError) as excp:
            _ = DirectionalArray.create_empty(10, directions=np.array([[1, 2, 3, 4], [1, 2, 3, 4]]))
        expected = 'dimensions for a DirectionalArray must be a 1D numpy array'
        self.assertEqual(str(excp.exception), expected)

        # Provided directions have values which are too close with respect to the direction_step
        with self.assertRaises(ValueError) as excp:
            _ = DirectionalArray.create_empty(10, directions=np.array([1, 2.50001, 3.4, 4]))
        expected = 'some dimensions values are too close to each other considering '
        expected += 'the dimensions quantization step: 1.0°'
        self.assertEqual(str(excp.exception), expected)

    def test_n_values_for(self) -> None:
        """ Test the values_for() method of DirectionalArray
        """
        # Array for this test.
        test_array = DirectionalArray(TEST_ARRAY1, np.array([4, -11, 100.]))

        # Retrieve an existing direction
        vector = test_array.values_for(-11)
        # FIXME: do we accept a 2D array as output for a single direction?
        self.assertEqual(vector.shape, (4, 1))
        self.assertEqual(vector.dtype, np.float64)
        self.assertTrue(np.array_equal(vector, np.array([[9.], [7.], [-35], [-1.7]])))
