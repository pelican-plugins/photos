import os
import re
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
        cls.settings["PHOTO_WATERMARK_TEXT"] = "watermark text"
        cls.settings["PHOTO_WATERMARK"] = True
        cls.output_path = cls.settings["OUTPUT_PATH"]
        photos.initialized(cls)

        context = cls.settings.copy()
        context["generated_content"] = {}
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
        photos.handle_signal_generator_init(cls.generator)
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
            list(self.get_article("photo").photo_image),
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
            list(self.get_article("filename").photo_image),
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
        assert len(photos.g_image_cache) == 5
        expected = [
            "photos/agallery/best",
            "photos/agallery/besta",
            "photos/agallery/bestt",
            "photos/agallery/night",
            "photos/agallery/nightt",
        ]
        assert sorted(expected) == sorted(photos.g_image_cache.keys())

    def test_inline_regex_simple(self):
        regex = re.compile(self.settings["PHOTO_INLINE_PATTERN"])
        m = regex.match("""<div gallery="foo"></div>""")
        assert m
        assert m.group("name") == "foo"
        assert m.group("type") == "gallery"

        parser = photos.HTMLTagParser()
        parser.feed(m.group())
        assert len(parser.tag_attrs) == 1
        assert parser.tag_attrs["gallery"] == "foo"

    def test_inline_regex_multi_element(self):
        regex = re.compile(self.settings["PHOTO_INLINE_PATTERN"])
        m = regex.match("""<div gallery="foo"><span test="more"></span></div>""")
        assert m
        assert m.group("name") == "foo"
        assert m.group("type") == "gallery"

        parser = photos.HTMLTagParser()
        parser.feed(m.group())
        assert len(parser.tag_attrs) == 2
        assert parser.tag_attrs["gallery"] == "foo"
        assert parser.tag_attrs["test"] == "more"

    def test_inline_regex_case(self):
        regex = re.compile(self.settings["PHOTO_INLINE_PATTERN"])
        m = regex.match("""<dIv foo="bar" gallery = "foo" bar = "foo"></DiV>""")
        assert m
        assert m.group("name") == "foo"
        assert m.group("type") == "gallery"

        parser = photos.HTMLTagParser()
        parser.feed(m.group())
        assert len(parser.tag_attrs) == 3
        assert parser.tag_attrs["gallery"] == "foo"
        assert parser.tag_attrs["foo"] == "bar"
        assert parser.tag_attrs["bar"] == "foo"

    def test_inline_regex_multi_line(self):
        regex = re.compile(self.settings["PHOTO_INLINE_PATTERN"])
        m = regex.match(
            """<dIv
        foo="bar"
        gallery = "foo"
        bar = "foo">
        </DiV>
        """
        )
        assert m
        assert m.group("name") == "foo"
        assert m.group("type") == "gallery"

        parser = photos.HTMLTagParser()
        parser.feed(m.group())
        assert len(parser.tag_attrs) == 3
        assert parser.tag_attrs["gallery"] == "foo"
        assert parser.tag_attrs["foo"] == "bar"
        assert parser.tag_attrs["bar"] == "foo"

    def test_inline_regex_short(self):
        regex = re.compile(self.settings["PHOTO_INLINE_PATTERN"])
        m = regex.match("""<div image="foo"/>""")
        assert m
        assert m.group("name") == "foo"
        assert m.group("type") == "image"

        parser = photos.HTMLTagParser()
        parser.feed(m.group())
        assert len(parser.tag_attrs) == 1
        assert parser.tag_attrs["image"] == "foo"


if __name__ == "__main__":
    unittest.main()
