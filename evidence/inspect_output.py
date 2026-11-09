"""Check the exported vectors and demonstrate an incorrect patch layout."""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
with np.load(ROOT / "assets/reference_outputs.npz", allow_pickle=False) as reference:
    ids = reference["ids"]
    expected = reference["output_tokens"]
    image = reference["preprocessed"][:1]
    first_patch = reference["patches"][0, 0]
with np.load(ROOT / "evidence/output_tokens.npz", allow_pickle=False) as result:
    actual = result["output_tokens"]
    np.testing.assert_array_equal(result["ids"], ids)
    assert actual.shape == (10, 512) and actual.dtype == np.float32
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-4)
    print("IDs:", ", ".join(ids))
    print("shape:", actual.shape, "dtype:", actual.dtype)
    print("first six:", np.array2string(actual[0, :6], precision=8))
    print(f"max norm error: {np.max(np.abs(np.linalg.norm(actual, axis=1)-1)):.3e}")

explicit = image[0, :, :32, :32].reshape(-1)
np.testing.assert_array_equal(explicit, first_patch)
wrong = image.reshape(1, 49, 3072)[0, 0]
print(f"direct reshape first patch max error: {np.max(np.abs(wrong-first_patch)):.3e}")
print(f"direct reshape unequal values: {np.count_nonzero(wrong != first_patch)}/3072")
