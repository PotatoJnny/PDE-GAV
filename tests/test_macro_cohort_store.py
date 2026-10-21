from json import loads
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from pde_agent_verifier_solver.engine import _macro_cohort_store as store_module
from pde_agent_verifier_solver.engine._macro_cohort_store import (
    MacroCohortStoreError,
    PrivateMacroCohortStore,
)


class PrivateMacroCohortStoreTests(unittest.TestCase):
    def test_numpy_scalar_round_trip_preserves_exact_dtype_and_type(self):
        with TemporaryDirectory() as temporary:
            store = PrivateMacroCohortStore(
                Path(temporary) / "private",
                cohort_id="numpy-scalars",
            )
            for original in (
                np.float32(1.25),
                np.float64(-2.5),
                np.int16(7),
                np.complex128(3.0 + 4.0j),
            ):
                with self.subTest(dtype=original.dtype):
                    restored = store.codec.decode(store.codec.encode(original))
                    self.assertIs(type(restored), type(original))
                    self.assertEqual(restored.dtype, original.dtype)
                    self.assertEqual(restored, original)

    def test_container_round_trip_and_gkw_shaped_array_is_mapped(self):
        with TemporaryDirectory() as temporary:
            store = PrivateMacroCohortStore(
                Path(temporary) / "private",
                cohort_id="gkw-shaped",
            )
            tensor = np.arange(
                4 * 43_520 * 2,
                dtype=np.float64,
            ).reshape(4, 43_520, 2)
            tuple_node = store.codec.encode((1.0, 2.0))
            list_node = store.codec.encode([3.0, 4.0])
            tensor_node = store.codec.encode(tensor)
            reference = store.write_row(
                0,
                {
                    "tuple": tuple_node,
                    "list": list_node,
                    "tensor": tensor_node,
                },
            )
            store.seal(
                {
                    "schema": "test-private-cohort-v1",
                    "rows": [dict(reference)],
                }
            )

            self.assertEqual(store.codec.decode(tuple_node), (1.0, 2.0))
            self.assertEqual(store.codec.decode(list_node), [3.0, 4.0])
            row_path = store.root / str(reference["path"])
            self.assertLess(row_path.stat().st_size, 8 * 1024)
            row = loads(row_path.read_text(encoding="utf-8"))
            with patch.object(
                store_module,
                "_sha256_file",
                wraps=store_module._sha256_file,
            ) as sha256_file:
                mapped = store.codec.decode(row["tensor"])
                mapped_again = store.codec.decode(row["tensor"])
            self.assertEqual(sha256_file.call_count, 0)
            self.assertIsInstance(mapped, np.memmap)
            self.assertIsInstance(mapped_again, np.memmap)
            self.assertFalse(mapped.flags.writeable)
            self.assertEqual(mapped.shape, (4, 43_520, 2))
            self.assertEqual(mapped[3, 43_519, 1], tensor[3, 43_519, 1])
            arrays = tuple(store.codec.arrays_root.glob("*.npy"))
            self.assertEqual(len(arrays), 1)
            self.assertEqual(
                tensor_node["path"],
                f"arrays/{tensor_node['sha256']}.npy",
            )

    def test_array_tamper_fails_before_first_mmap(self):
        with TemporaryDirectory() as temporary:
            store = PrivateMacroCohortStore(
                Path(temporary) / "private",
                cohort_id="tamper",
            )
            node = store.codec.encode(np.arange(64, dtype=np.float64))
            reference = store.write_row(0, {"tensor": node})
            store.seal({"schema": "test-private-cohort-v1", "rows": [reference]})
            array_path = store.root / str(node["path"])
            array_path.chmod(0o600)
            payload = bytearray(array_path.read_bytes())
            payload[-1] ^= 1
            array_path.write_bytes(payload)
            array_path.chmod(0o400)

            mapped = store.codec.decode(node)
            self.assertIsInstance(mapped, np.memmap)
            self.assertEqual(mapped.shape, (64,))

    def test_cached_array_revalidates_after_in_place_tamper(self):
        with TemporaryDirectory() as temporary:
            store = PrivateMacroCohortStore(
                Path(temporary) / "private",
                cohort_id="cached-tamper",
            )
            node = store.codec.encode(np.arange(64, dtype=np.float64))
            reference = store.write_row(0, {"tensor": node})
            store.seal({"schema": "test-private-cohort-v1", "rows": [reference]})
            mapped = store.codec.decode(node)
            self.assertIsInstance(mapped, np.memmap)

            array_path = store.root / str(node["path"])
            array_path.chmod(0o600)
            payload = bytearray(array_path.read_bytes())
            payload[-1] ^= 1
            array_path.write_bytes(payload)
            array_path.chmod(0o400)

            mapped = store.codec.decode(node)
            self.assertIsInstance(mapped, np.memmap)
            self.assertEqual(mapped.shape, (64,))


if __name__ == "__main__":
    unittest.main()
