import os
from shutil import rmtree
from tempfile import mkdtemp

from pelican.generators import ArticlesGenerator
from pelican.tests.support import get_settings, unittest

import photos

CUR_DIR = os.path.dirname(__file__)


class TestPhotos(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_path = mkdtemp(prefix="pelicantests.")
        cls.settings = get_settings(filenames={})
        cls.settings["PATH"] = os.path.join(CUR_DIR, "test_data")
        cls.settings["PHOTO_LIBRARY"] = os.path.join(CUR_DIR, "test_data")
        cls.settings["DEFAULT_DATE"] = (1970, 1, 1)
        cls.settings["FILENAME_METADATA"] = "(?P<slug>[^.]+)"
        cls.settings["PLUGINS"] = [photos]
        cls.settings["CACHE_CONTENT"] = False
        cls.settings["OUTPUT_PATH"] = cls.temp_path
        cls.settings["SITEURL"] = "http://getpelican.com/sub"
        cls.settings["AUTHOR"] = "Bob Anonymous"
        cls.output_path = cls.settings["OUTPUT_PATH"]
        photos.initialized(cls)

        context = cls.settings.copy()
        context["generated_content"] = dict()
        context["static_links"] = set()
        context["static_content"] = {}
        cls.generator = ArticlesGenerator(
            context=context,
            settings=cls.settings,
            path=cls.settings["PATH"],
            theme=cls.settings["THEME"],
            output_path=cls.settings["OUTPUT_PATH"],
        )
        photos.register()
        cls.generator.generate_context()
        photos.handle_signal_all_generators_finalized([cls.generator])

    @classmethod
    def tearDownClass(cls):
        rmtree(cls.temp_path)

    def test_image(self):
        for a in self.generator.articles:
            if "image" in a.metadata:
                self.assertTrue(
                    hasattr(a, "photo_image"),
                    msg="{} not recognized.".format(a.metadata["image"]),
                )

    def test_gallery(self):
        for a in self.generator.articles:
            if "gallety" in a.metadata:
                self.assertTrue(
                    hasattr(a, "photo_gallery"),
                    msg="{} not recognized.".format(a.metadata["gallery"]),
                )

    def get_article(self, slug):
        for a in self.generator.articles:
            if slug == a.slug:
                return a
        return None

    def test_photo_article_image(self):
        self.assertEqual(
            [v for v in self.get_article("photo").photo_image],
            ["best.jpg", "photos/agallery/besta.jpg", "photos/agallery/bestt.jpg"],
        )

    def test_photo_article_gallery(self):
        photo_gallery = self.get_article("filename").photo_gallery[0][1]
        self.assertEqual(
            [str(v) for v in photo_gallery[0]],
            [
                "best.jpg",
                "photos/agallery/best.jpg",
                "photos/agallery/bestt.jpg",
                "EXIF-best",
                "Caption-best",
            ],
        )
        self.assertEqual(
            [str(v) for v in photo_gallery[1]],
            [
                "night.png",
                "photos/agallery/night.jpg",
                "photos/agallery/nightt.jpg",
                "EXIF-night",
                "",
            ],
        )

    def test_photo_article_body(self):
        expected = (
            "<p>Here is my best photo, again.</p>\n"
            "<p>"
            '<img alt="" src="http://getpelican.com/sub/photos/agallery/besta.jpg">.'
            "</p>"
        )
        self.assertEqual(expected, self.get_article("photo").content)

    def test_filename_article_image(self):
        self.assertEqual(
            ["best.jpg", "photos/agallery/besta.jpg", "photos/agallery/bestt.jpg"],
            [v for v in self.get_article("filename").photo_image],
        )

    def test_filename_article_gallery(self):
        photo_gallery = self.get_article("filename").photo_gallery[0][1]
        self.assertEqual(
            [str(v) for v in photo_gallery[0]],
            [
                "best.jpg",
                "photos/agallery/best.jpg",
                "photos/agallery/bestt.jpg",
                "EXIF-best",
                "Caption-best",
            ],
        )
        self.assertEqual(
            [str(v) for v in photo_gallery[1]],
            [
                "night.png",
                "photos/agallery/night.jpg",
                "photos/agallery/nightt.jpg",
                "EXIF-night",
                "",
            ],
        )

    def test_filename_article_body(self):
        expected = (
            "<p>Here is my best photo, again.</p>\n"
            '<p><img alt="" src="{filename}agallery/best.jpg">.</p>'
        )
        self.assertEqual(expected, self.get_article("filename").content)

    def test_queue_resize(self):
        assert len(photos.DEFAULT_CONFIG["queue_resize"]) == 5
        expected = [
            "photos/agallery/best",
            "photos/agallery/besta",
            "photos/agallery/bestt",
            "photos/agallery/night",
            "photos/agallery/nightt",
        ]
        assert sorted(expected) == sorted(photos.DEFAULT_CONFIG["queue_resize"].keys())


if __name__ == "__main__":
    unittest.main()
