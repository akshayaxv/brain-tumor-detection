import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import brain_tumor_detection as btd


def test_model_output_shape():
    model = btd.build_model(weights=None)
    assert model.output_shape[-1] == 2
