# -*- coding: utf-8 -*-
""" Definition of the ConstrainedDict class

:authors: see AUTHORS file
:organization: CNES, LEGOS, SHOM
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 12 oct 2024

  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
  in compliance with the License. You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License
  is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
  or implied. See the License for the specific language governing permissions and
  limitations under the License.
"""
from abc import ABC, abstractmethod
from typing import Any  # @NoMove


class ConstrainedDict(ABC, dict):
    """ An abstract dictionary whose keys and values can be checked and possibly modified
    to suit specific constraints before insertion. The constraints are implemented by 2 abstract
    methods which takes either the key or value as argument and return the key or value to
    insert into the dictionary.
    """

    def __setitem__(self, key: Any, value: Any) -> None:
        new_key = self.constrained_key(key)
        new_value = self.constrained_value(value)
        dict.__setitem__(self, new_key, new_value)

    def __getitem__(self, key: Any) -> Any:
        try:
            return dict.__getitem__(self, key)
        except KeyError:
            return dict.__getitem__(self, self.constrained_key(key))

    @abstractmethod
    def constrained_key(self, key: Any) -> Any:
        """ Returns the key to use in this dictionary which satisfies some client rules.

        :param key: the key to check and use to generate a constrained key
        :returns: the constrained key to use in the dictionary.
        :raises KeyError: when the input key cannot be constrained due to its type or value.
        """

    @abstractmethod
    def constrained_value(self, value: Any) -> Any:
        """ Returns the value to store in this dictionary which satisfies some client rules.

        :param value: the value to check and use to generate a constrained value
        :returns: the constrained value to store in the dictionary.
        :raises ValueError: when the input value cannot be constrained due to its type or value.
        """
