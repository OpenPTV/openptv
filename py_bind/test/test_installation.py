import unittest

class TestSingleModule(unittest.TestCase):
    def test_tracking_framebuf_import(self):
        """Test that tracking_framebuf module can be imported"""
        try:
            import optv.tracking_framebuf
            self.assertTrue(True, "tracking_framebuf module imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import tracking_framebuf: {str(e)}")

if __name__ == '__main__':
    unittest.main(verbosity=2)
