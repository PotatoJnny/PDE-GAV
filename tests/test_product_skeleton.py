import unittest

from pde_agent_verifier_solver import __version__
from pde_agent_verifier_solver import app, engine, panel, plugins, sdk


class ProductSkeletonTests(unittest.TestCase):
    def test_product_package_and_boundaries_are_importable(self) -> None:
        self.assertEqual(__version__, "0.1.0.dev0")
        self.assertEqual(len({app, sdk, engine, panel, plugins}), 5)


if __name__ == "__main__":
    unittest.main()
