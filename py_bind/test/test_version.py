import unittest
import optv

class TestVersion(unittest.TestCase):
    def test_version_available(self):
        """Test that version is available and properly formatted."""
        self.assertTrue(hasattr(optv, '__version__'))
        self.assertIsInstance(optv.__version__, str)
        # Check version format (x.y.z)
        parts = optv.__version__.split('.')
        self.assertEqual(len(parts), 3)
        self.assertTrue(all(part.isdigit() for part in parts))

if __name__ == '__main__':
    unittest.main()