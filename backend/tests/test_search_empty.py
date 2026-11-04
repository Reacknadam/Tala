"""Regression test: search on an empty index must not crash.

``search_by_embedding`` documents a ``(hits, scanned)`` tuple, but the
empty-index branch historically returned a bare list.
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from app.vision import search


class EmptyIndexTests(unittest.TestCase):
    def test_empty_index_returns_empty_tuple(self):
        with patch.object(search.storage, "fetch_candidate_detections", return_value=[]):
            hits, scanned = search.search_by_embedding(np.zeros(8, dtype=np.float32))
        self.assertEqual(hits, [])
        self.assertEqual(scanned, 0)


if __name__ == "__main__":
    unittest.main()
