import datetime
import itertools
import json
import logging
import mimetypes
import multiprocessing
import os
import pprint
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import urllib.parse

from pelican import Pelican, signals
from pelican.contents import Article, Page
import pelican.generators
from pelican.generators import ArticlesGenerator, PagesGenerator
from pelican.settings import DEFAULT_CONFIG
from pelican.utils import pelican_open

logger = logging.getLogger(__name__)
pelican_settings: Dict[str, Any] = {}
pelican_output_path: Optional[str] = None

try:
    from PIL import Image as PILImage
    from PIL import ImageDraw, ImageEnhance, ImageFont, ImageOps
except ImportError as e:
    logger.error("PIL/Pillow not found")
    raise e

try:
    import piexif
except ImportError:
    ispiexif = False
    logger.warning("piexif not found! Cannot use exif manipulation features")
else:
    ispiexif = True
    logger.debug("piexif found.")


class InternalError(Exception):
    pass


class FileNotFound(Exception):
    pass


class GalleryNotFound(Exception):
    pass


class ImageExcluded(Exception):
    pass


class ArticleImage:
    def __init__(
        self,
        content: pelican.contents.Content,
        filename: str,
        generator: pelican.generators.Generator,
    ):
        """
        Images/photos on the top of an article or page.
        """
        self._filename = filename
        if filename.startswith("{photo}"):
            path = os.path.join(
                os.path.expanduser(pelican_settings["PHOTO_LIBRARY"]),
                image_clipper(filename),
            )
            image = image_clipper(filename)
        elif filename.startswith("{filename}"):
            path = os.path.join(
                generator.path, content.relative_dir, file_clipper(filename)
            )
            image = file_clipper(filename)

        if not os.path.isfile(path):
            raise FileNotFound(f"No photo for {content.source_path} at {path}")

        photo = os.path.splitext(image)[0].lower() + "a"
        thumb = os.path.splitext(image)[0].lower() + "t"
        img = Image(
            src=path,
            dst=os.path.join("photos", photo),
            specs=pelican_settings["PHOTO_ARTICLE"],
        )
        self.image = enqueue_resize(img)
        img = Image(
            src=path,
            dst=os.path.join("photos", thumb),
            specs=pelican_settings["PHOTO_THUMB"],
        )
        self.thumb = enqueue_resize(img)
        self.file = os.path.basename(image).lower()

    def __getitem__(self, index):
        """
        Legacy support
        """
        if index == 0:
            return self.file
        elif index == 1:
            return self.image.web_filename
        elif index == 2:
            return self.thumb.web_filename
        else:
            raise IndexError


class BaseNoteCache:
    note_cache: Dict[str, "BaseNoteCache"] = {}
    note_filename = None

    def __init__(self, filename):
        self.filename = filename
        self.notes: Dict[str, str] = {}
        self._read()

    def _read(self):
        try:
            with pelican_open(self.filename) as text:
                for line in text.splitlines():
                    if line.startswith("#"):
                        continue
                    m = line.split(":", 1)
                    if len(m) > 1:
                        pic = m[0].strip()
                        note = m[1].strip()
                        if pic and note:
                            self.notes[pic] = note
                    else:
                        self.notes[line] = ""
        except Exception as e:
            logger.debug(
                f"photos: read_notes issue at file {self.filename}. "
                f"Debug message:{e}"
            )

    def get_value(self, source_image):
        return self.notes.get(os.path.basename(source_image.filename))

    @classmethod
    def from_cache(cls, source_image: "SourceImage"):
        filename = os.path.join(
            os.path.dirname(source_image.filename), cls.note_filename
        )
        notes = cls.note_cache.get(filename)
        if notes is None:
            notes = cls(filename=filename)
        return notes


class CaptionCache(BaseNoteCache):
    note_filename = "captions.txt"


class ExifCache(BaseNoteCache):
    note_filename = "exif.txt"


class ExcludeCache(BaseNoteCache):
    note_filename = "blacklist.txt"


class BaseNote:
    cache_class = None

    def __init__(self, source_image):
        print(self.cache_class)
        self.cache = self.cache_class.from_cache(source_image)
        self._value = self.cache.get_value(source_image)

    def __str__(self):
        return self.value

    @property
    def value(self):
        return self._value


class Caption(BaseNote):
    cache_class = CaptionCache


class Exif(BaseNote):
    cache_class = ExifCache


class ExcludeList(BaseNote):
    cache_class = ExcludeCache


class ContentImage:
    def __init__(self, filename):
        self.filename = filename
        self._src_filename = os.path.join(
            os.path.expanduser(pelican_settings["PHOTO_LIBRARY"]), self.filename
        )

        if not os.path.isfile(self._src_filename):
            raise FileNotFound(f"No photo for {self._src_filename}")

        img = Image(
            src=self._src_filename,
            dst=os.path.join("photos", os.path.splitext(filename)[0].lower() + "a"),
            specs=pelican_settings["PHOTO_ARTICLE"],
        )
        self.image = enqueue_resize(img)


class ContentImageLightbox:
    def __init__(self, filename):
        self.filename = filename
        self._src_filename = os.path.join(
            os.path.expanduser(pelican_settings["PHOTO_LIBRARY"]), self.filename
        )

        if not os.path.isfile(self._src_filename):
            raise FileNotFound(f"No photo for {self._src_filename}")

        img = Image(
            src=self._src_filename,
            dst=os.path.join("photos", os.path.splitext(filename)[0].lower()),
            specs=pelican_settings["PHOTO_GALLERY"],
        )
        self.image = enqueue_resize(img)

        img = Image(
            src=self._src_filename,
            dst=os.path.join("photos", os.path.splitext(filename)[0].lower() + "t"),
            specs=pelican_settings["PHOTO_THUMB"],
        )
        self.thumb = enqueue_resize(img)

    @property
    def caption(self):
        return self.image.caption


class Gallery:
    def __init__(self, content: Union[Article, Page], location_parsed):
        """
        Process a single gallery

        - look for images
        - read meta data
        - read exif data
        - enqueue the images to be processed
        """
        self.content = content

        if location_parsed["type"] == "{photo}":
            dir_gallery = os.path.join(
                os.path.expanduser(pelican_settings["PHOTO_LIBRARY"]),
                location_parsed["location"],
            )
            rel_gallery = location_parsed["location"]
        elif location_parsed["type"] == "{filename}":
            base_path = os.path.join(pelican_settings["PATH"], content.relative_dir)
            dir_gallery = os.path.join(base_path, location_parsed["location"])
            rel_gallery = os.path.join(
                content.relative_dir, location_parsed["location"]
            )

        if not os.path.isdir(dir_gallery):
            raise GalleryNotFound(
                "Gallery does not exist: {} at {}".format(
                    location_parsed["location"], dir_gallery
                )
            )

        self.src_dir = dir_gallery

        logger.info(f"photos: Gallery detected: {rel_gallery}")
        self.dst_dir = os.path.join("photos", rel_gallery.lower())
        self.images: List[GalleryImage] = []

        self.title = location_parsed["title"]
        for pic in sorted(os.listdir(dir_gallery)):
            if pic.startswith("."):
                continue
            if pic.endswith(".txt"):
                continue

            try:
                self.images.append(GalleryImage(filename=pic, gallery=self))
            except ImageExcluded:
                logger.debug(f"photos: Image {pic} excluded")

    def __getitem__(self, item):
        if item == 0:
            return self.title
        elif item == 1:
            return self.images
        else:
            raise IndexError


class GalleryImage:
    def __init__(self, filename, gallery: Gallery):
        self._gallery = gallery
        self.filename = filename

        img = Image(
            src=os.path.join(self._gallery.src_dir, self.filename),
            dst=os.path.join(
                self._gallery.dst_dir, os.path.splitext(filename)[0].lower()
            ),
            specs=pelican_settings["PHOTO_GALLERY"],
        )
        if img.is_excluded:
            raise ImageExcluded("Image excluded from gallery")
        self.image = enqueue_resize(img)

        img = Image(
            src=os.path.join(self._gallery.src_dir, self.filename),
            dst=os.path.join(
                self._gallery.dst_dir, os.path.splitext(filename)[0].lower() + "t"
            ),
            specs=pelican_settings["PHOTO_THUMB"],
        )
        self.thumb = enqueue_resize(img)

    def __getitem__(self, item):
        """
        Legacy support
        """
        if item == 0:
            return self.filename
        elif item == 1:
            return self.image
        elif item == 2:
            return self.thumb
        elif item == 3:
            if self.exif is None:
                return ""
            return self.exif.value
        elif item == 4:
            if self.caption is None:
                return ""
            return self.caption.value

        raise IndexError

    @property
    def caption(self):
        return self.image.caption

    @property
    def exif(self):
        return self.image.exif

    @property
    def is_excluded(self):
        return self.image.is_excluded


class Image:
    def __init__(
        self,
        src,
        dst,
        spec: Optional[Dict[str, Any]] = None,
        specs: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        if spec is not None and specs is not None:
            raise ValueError("Both spec and specs must not be provided")

        self.source_image = SourceImage.from_cache(src)
        self.dst = dst

        self._height: Optional[int] = None
        self._width: Optional[int] = None

        if spec is None:
            if specs is None:
                raise ValueError("Only one of spec and specs must be provided")
            spec = specs.get(self.source_image.type)
            if spec is None:
                spec = specs["default"]

        self.spec: Dict[str, Any] = spec.copy()
        self.type = spec["type"].lower()

        image_options: Dict[str, Any] = pelican_settings[
            "PHOTO_DEFAULT_IMAGE_OPTIONS"
        ].get(self.type)
        if image_options is None:
            image_options = {}
        if not isinstance(image_options, dict):
            logger.warning(
                f"photos: Wrong type for default image options for type {spec['type']}"
            )
            image_options = {}
        image_options = image_options.copy()
        image_options.update(spec.get("options", {}))
        self.spec["options"] = image_options

        srcset_specs: Optional[List, Tuple] = self.spec.get("srcset")
        if not isinstance(srcset_specs, (list, tuple)):
            srcset_specs = []

        self.srcset = ImageSrcSet()
        for srcset_spec in srcset_specs:
            img = SrcSetImage(
                src=self.source_image.filename, dst=self.dst, spec=srcset_spec
            )
            self.srcset.append(enqueue_resize(img))

        self.output_filename = "{filename}.{extension}".format(
            filename=os.path.join(pelican_output_path, self.dst),
            extension=pelican_settings["PHOTO_FILE_EXTENSIONS"].get(
                self.type, self.type
            ),
        )

        self.web_filename = "{resized}.{extension}".format(
            resized=self.dst,
            extension=pelican_settings["PHOTO_FILE_EXTENSIONS"].get(
                self.type, self.type
            ),
        )

    def __str__(self):
        return self.web_filename

    @property
    def caption(self):
        return self.source_image.caption

    @property
    def exif(self):
        return self.source_image.exif

    @property
    def is_excluded(self):
        return self.source_image.is_excluded

    @staticmethod
    def is_alpha(img):
        return (
            True
            if img.mode in ("RGBA", "LA")
            or (img.mode == "P" and "transparency" in img.info)
            else False
        )

    @property
    def height(self):
        if self._height is None:
            self._load_result_info()
        return self._height

    @property
    def width(self):
        if self._width is None:
            self._load_result_info()
        return self._width

    def _load_result_info(self):
        img: PILImage.Image = PILImage.open(self.output_filename)
        self._height = img.height
        self._width = img.width

    def apply_result_info(self, info: Dict[str, Any]):
        allowed_names = ("_height", "_width")
        for name in allowed_names:
            if name in info:
                setattr(self, name, info[name])

    def manipulate_exif(self, img):
        try:
            exif = piexif.load(img.info["exif"])
        except Exception:
            logger.debug("EXIF information not found")
            exif = {}

        if pelican_settings["PHOTO_EXIF_AUTOROTATE"]:
            img, exif = self.rotate(img, exif)

        if pelican_settings["PHOTO_EXIF_REMOVE_GPS"]:
            exif.pop("GPS")

        if pelican_settings["PHOTO_EXIF_COPYRIGHT"]:
            # Be minimally destructive to any preset EXIF author or copyright
            # information. If there is copyright or author information, prefer that
            # over everything else.
            if not exif["0th"].get(piexif.ImageIFD.Artist):
                exif["0th"][piexif.ImageIFD.Artist] = pelican_settings[
                    "PHOTO_EXIF_COPYRIGHT_AUTHOR"
                ]
                author = pelican_settings["PHOTO_EXIF_COPYRIGHT_AUTHOR"]

            if not exif["0th"].get(piexif.ImageIFD.Copyright):
                license = build_license(
                    pelican_settings["PHOTO_EXIF_COPYRIGHT"], author
                )
                exif["0th"][piexif.ImageIFD.Copyright] = license

        return img, piexif.dump(exif)

    def process(self, key):
        self.resize()
        return key, {"_height": self._height, "_width": self._width}

    def reduce_opacity(self, im, opacity):
        """Reduces Opacity.

        Returns an image with reduced opacity.
        Taken from http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/362879
        """
        assert opacity >= 0 and opacity <= 1
        if self.is_alpha(im):
            im = im.copy()
        else:
            im = im.convert("RGBA")

        alpha = im.split()[3]
        alpha = ImageEnhance.Brightness(alpha).enhance(opacity)
        im.putalpha(alpha)
        return im

    @staticmethod
    def remove_alpha(img, bg_color):
        background = PILImage.new("RGB", img.size, bg_color)
        background.paste(img, mask=img.split()[3])  # 3 is the alpha channel
        return background

    def resize(self):
        spec = self.spec

        process = multiprocessing.current_process()
        logger.info(
            f"photos: make photo(PID: {process.pid}) {self.source_image.filename} "
            f"-> {self.output_filename}"
        )

        if os.path.isfile(self.output_filename) and os.path.getmtime(
            self.source_image.filename
        ) <= os.path.getmtime(self.output_filename):
            logger.debug(
                f"Skipping orig: {os.path.getmtime(self.source_image.filename)} "
                f"{os.path.getmtime(self.output_filename)}"
            )
            self._load_result_info()
            return

        im = self.source_image.open()

        # if (
        #     ispiexif and pelican_settings["PHOTO_EXIF_KEEP"] and im.format == "JPEG"
        # ):  # Only works with JPEG exif for sure.
        #     try:
        #         im, exif_copy = manipulate_exif(im)
        #     except Exception:
        #         logger.info(f"photos: no EXIF or EXIF error in {orig}")
        #         exif_copy = b""
        # else:
        #     exif_copy = b""
        #
        # icc_profile = im.info.get("icc_profile", None)

        if (
            pelican_settings["PHOTO_SQUARE_THUMB"]
            and spec == pelican_settings["PHOTO_THUMB"]
        ):
            im = ImageOps.fit(im, (spec["width"], spec["height"]), PILImage.ANTIALIAS)

        im.thumbnail((spec["width"], spec["height"]), PILImage.ANTIALIAS)

        if self.is_alpha(im):
            im = self.remove_alpha(im, pelican_settings["PHOTO_ALPHA_BACKGROUND_COLOR"])

        directory = os.path.dirname(self.output_filename)
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except Exception:
                logger.exception(f"Could not create {directory}")
        else:
            logger.debug(f"Directory already exists at {os.path.split(directory)[0]}")

        if pelican_settings["PHOTO_WATERMARK"]:
            isthumb = True if spec == pelican_settings["PHOTO_THUMB"] else False
            if not isthumb or (isthumb and pelican_settings["PHOTO_WATERMARK_THUMB"]):
                im = self.watermark(im)

        image_options = spec.get("options", {})
        im.save(
            self.output_filename,
            self.type,
            # icc_profile=icc_profile,
            # exif=exif_copy,
            **image_options,
        )
        self._height = im.height
        self._width = im.width

    @staticmethod
    def rotate(img, exif_dict):
        if "exif" in img.info and piexif.ImageIFD.Orientation in exif_dict["0th"]:
            orientation = exif_dict["0th"].pop(piexif.ImageIFD.Orientation)
            if orientation == 2:
                img = img.transpose(PILImage.FLIP_LEFT_RIGHT)
            elif orientation == 3:
                img = img.rotate(180)
            elif orientation == 4:
                img = img.rotate(180).transpose(PILImage.FLIP_LEFT_RIGHT)
            elif orientation == 5:
                img = img.rotate(-90).transpose(PILImage.FLIP_LEFT_RIGHT)
            elif orientation == 6:
                img = img.rotate(-90, expand=True)
            elif orientation == 7:
                img = img.rotate(90).transpose(PILImage.FLIP_LEFT_RIGHT)
            elif orientation == 8:
                img = img.rotate(90)

        return img, exif_dict

    def watermark(self, image):
        margin = [10, 10]
        opacity = 0.6

        watermark_layer = PILImage.new("RGBA", image.size, (0, 0, 0, 0))
        draw_watermark = ImageDraw.Draw(watermark_layer)
        text_reducer = 32
        image_reducer = 8
        text_size = [0, 0]
        mark_size = [0, 0]
        text_position = [0, 0]

        if pelican_settings["PHOTO_WATERMARK_TEXT"]:
            font_name = "SourceCodePro-Bold.otf"
            default_font = os.path.join(DEFAULT_CONFIG["plugin_dir"], font_name)
            font = ImageFont.FreeTypeFont(
                default_font, watermark_layer.size[0] // text_reducer
            )
            text_size = draw_watermark.textsize(
                pelican_settings["PHOTO_WATERMARK_TEXT"], font
            )
            text_position = [image.size[i] - text_size[i] - margin[i] for i in [0, 1]]
            draw_watermark.text(
                text_position,
                pelican_settings["PHOTO_WATERMARK_TEXT"],
                pelican_settings["PHOTO_WATERMARK_TEXT_COLOR"],
                font=font,
            )

        if pelican_settings["PHOTO_WATERMARK_IMG"]:
            mark_image = PILImage.open(pelican_settings["PHOTO_WATERMARK_IMG"])
            mark_image_size = [
                watermark_layer.size[0] // image_reducer for size in mark_size
            ]
            mark_image_size = (
                pelican_settings["PHOTO_WATERMARK_IMG_SIZE"]
                if pelican_settings["PHOTO_WATERMARK_IMG_SIZE"]
                else mark_image_size
            )
            mark_image.thumbnail(mark_image_size, PILImage.ANTIALIAS)
            mark_position = [
                watermark_layer.size[i] - mark_image.size[i] - margin[i] for i in [0, 1]
            ]
            mark_position = tuple(
                [
                    mark_position[0] - (text_size[0] // 2) + (mark_image_size[0] // 2),
                    mark_position[1] - text_size[1],
                ]
            )

            if not self.is_alpha(mark_image):
                mark_image = mark_image.convert("RGBA")

            watermark_layer.paste(mark_image, mark_position, mark_image)

        watermark_layer = self.reduce_opacity(watermark_layer, opacity)
        image.paste(watermark_layer, (0, 0), watermark_layer)

        return image


class SrcSetImage(Image):
    def __init__(
        self,
        src,
        dst,
        spec: Optional[Dict[str, Any]] = None,
    ):
        self.descriptor = spec.get("srcset_descriptor", f"{spec['width']}w")

        dst_suffix = spec.get("srcset_extension")
        if dst_suffix is None:
            dst_suffix = self.descriptor

        dst = f"{dst}_{dst_suffix}"
        super().__init__(src=src, dst=dst, spec=spec)


class ImageSrcSet(list):
    @property
    def html_srcset(self):
        items = []
        img: SrcSetImage
        for img in self:
            items.append(
                "{url} {descriptor}".format(
                    url=urllib.parse.urljoin(
                        pelican_settings["SITEURL"], img.web_filename
                    ),
                    descriptor=img.descriptor,
                )
            )
        return ", ".join(items)


class SourceImage:
    image_cache: Dict[str, "SourceImage"] = {}

    def __init__(self, filename):
        self.filename = filename
        self.mimetype, _ = mimetypes.guess_type(filename)
        _, _, image_type = self.mimetype.partition("/")
        self.type = image_type.lower()

        self._caption = Caption(source_image=self)
        self._exif = Exif(source_image=self)
        self._excluded = ExcludeList(source_image=self)

    @property
    def caption(self):
        if self._caption.value is None:
            return None
        return self._caption

    @property
    def exif(self):
        if self._exif.value is None:
            return None
        return self._exif

    @property
    def is_excluded(self):
        if self._excluded.value is None:
            return False
        return True

    def open(self):
        logger.debug(f"photos: Open file {self.filename}")
        return PILImage.open(self.filename)

    @classmethod
    def from_cache(cls, filename):
        source_image = cls.image_cache.get(filename)
        if source_image is None:
            source_image = cls(filename=filename)
            cls.image_cache[filename] = source_image

        return source_image


def initialized(pelican: Pelican):
    p = os.path.expanduser("~/Pictures")

    DEFAULT_CONFIG.setdefault("PHOTO_LIBRARY", p)
    DEFAULT_CONFIG.setdefault("PHOTO_GALLERY", (1024, 768, 80))
    DEFAULT_CONFIG.setdefault("PHOTO_ARTICLE", (760, 506, 80))
    DEFAULT_CONFIG.setdefault("PHOTO_THUMB", (192, 144, 60))
    DEFAULT_CONFIG.setdefault("PHOTO_SQUARE_THUMB", False)
    DEFAULT_CONFIG.setdefault("PHOTO_GALLERY_TITLE", "")
    DEFAULT_CONFIG.setdefault("PHOTO_ALPHA_BACKGROUND_COLOR", (255, 255, 255))
    DEFAULT_CONFIG.setdefault("PHOTO_WATERMARK", False)
    DEFAULT_CONFIG.setdefault("PHOTO_WATERMARK_THUMB", False)
    DEFAULT_CONFIG.setdefault("PHOTO_WATERMARK_TEXT", DEFAULT_CONFIG["SITENAME"])
    DEFAULT_CONFIG.setdefault("PHOTO_WATERMARK_TEXT_COLOR", (255, 255, 255))
    DEFAULT_CONFIG.setdefault("PHOTO_WATERMARK_IMG", "")
    DEFAULT_CONFIG.setdefault("PHOTO_WATERMARK_IMG_SIZE", False)
    DEFAULT_CONFIG.setdefault("PHOTO_RESIZE_JOBS", 1)
    DEFAULT_CONFIG.setdefault("PHOTO_EXIF_KEEP", False)
    DEFAULT_CONFIG.setdefault("PHOTO_EXIF_REMOVE_GPS", False)
    DEFAULT_CONFIG.setdefault("PHOTO_EXIF_AUTOROTATE", True)
    DEFAULT_CONFIG.setdefault("PHOTO_EXIF_COPYRIGHT", False)
    DEFAULT_CONFIG.setdefault("PHOTO_EXIF_COPYRIGHT_AUTHOR", DEFAULT_CONFIG["SITENAME"])
    DEFAULT_CONFIG.setdefault("PHOTO_LIGHTBOX_GALLERY_ATTR", "data-lightbox")
    DEFAULT_CONFIG.setdefault("PHOTO_LIGHTBOX_CAPTION_ATTR", "data-title")

    DEFAULT_CONFIG["queue_resize"] = {}
    DEFAULT_CONFIG["created_galleries"] = {}
    DEFAULT_CONFIG["plugin_dir"] = os.path.dirname(os.path.realpath(__file__))

    if pelican:
        pelican.settings.setdefault("PHOTO_LIBRARY", p)
        pelican.settings.setdefault("PHOTO_GALLERY", (1024, 768, 80))
        pelican.settings.setdefault("PHOTO_ARTICLE", (760, 506, 80))
        pelican.settings.setdefault("PHOTO_THUMB", (192, 144, 60))
        pelican.settings.setdefault("PHOTO_SQUARE_THUMB", False)
        pelican.settings.setdefault("PHOTO_GALLERY_TITLE", "")
        pelican.settings.setdefault("PHOTO_ALPHA_BACKGROUND_COLOR", (255, 255, 255))
        pelican.settings.setdefault("PHOTO_WATERMARK", False)
        pelican.settings.setdefault("PHOTO_WATERMARK_THUMB", False)
        pelican.settings.setdefault(
            "PHOTO_WATERMARK_TEXT", pelican.settings["SITENAME"]
        )
        pelican.settings.setdefault("PHOTO_WATERMARK_TEXT_COLOR", (255, 255, 255))
        pelican.settings.setdefault("PHOTO_WATERMARK_IMG", "")
        pelican.settings.setdefault("PHOTO_WATERMARK_IMG_SIZE", False)
        pelican.settings.setdefault("PHOTO_RESIZE_JOBS", 1)
        pelican.settings.setdefault("PHOTO_EXIF_KEEP", False)
        pelican.settings.setdefault("PHOTO_EXIF_REMOVE_GPS", False)
        pelican.settings.setdefault("PHOTO_EXIF_AUTOROTATE", True)
        pelican.settings.setdefault("PHOTO_EXIF_COPYRIGHT", False)
        pelican.settings.setdefault(
            "PHOTO_EXIF_COPYRIGHT_AUTHOR", pelican.settings["AUTHOR"]
        )
        pelican.settings.setdefault(
            "PHOTO_FILE_EXTENSIONS", {"jpeg": "jpg", "webp": "webp"}
        )
        pelican.settings.setdefault("PHOTO_LIGHTBOX_GALLERY_ATTR", "data-lightbox")
        pelican.settings.setdefault("PHOTO_LIGHTBOX_CAPTION_ATTR", "data-title")

        pelican.settings.setdefault("PHOTO_INLINE_GALLERY_ENABLED", False)
        pelican.settings.setdefault(
            "PHOTO_INLINE_GALLERY_PATTERN", r"gallery::(?P<gallery_name>[/{}\w_-]+)"
        )
        pelican.settings.setdefault("PHOTO_INLINE_GALLERY_TEMPLATE", "inline_gallery")
        pelican.settings.setdefault(
            "PHOTO_DEFAULT_IMAGE_OPTIONS", {"jpeg": {"optimize": True}}
        )

    global pelican_settings
    pelican_settings = pelican.settings
    global pelican_output_path
    pelican_output_path = pelican.output_path


def enqueue_resize(img: Image) -> Image:
    if img.dst not in DEFAULT_CONFIG["queue_resize"]:
        DEFAULT_CONFIG["queue_resize"][img.dst] = img
    elif (
        DEFAULT_CONFIG["queue_resize"][img.dst].source_image != img.source_image
        or DEFAULT_CONFIG["queue_resize"][img.dst].spec != img.spec
    ):
        raise InternalError(
            "resize conflict for {}, {}-{} is not {}-{}".format(
                img.dst,
                DEFAULT_CONFIG["queue_resize"][img.dst].source_image.filename,
                DEFAULT_CONFIG["queue_resize"][img.dst].spec,
                img.source_image.filename,
                img.spec,
            )
        )
    return img


def build_license(license, author):
    year = datetime.datetime.now().year
    license_file = os.path.join(DEFAULT_CONFIG["plugin_dir"], "licenses.json")

    with open(license_file) as data_file:
        licenses = json.load(data_file)

    if any(license in k for k in licenses):
        return licenses[license]["Text"].format(
            Author=author, Year=year, URL=licenses[license]["URL"]
        )
    else:
        return "Copyright {Year} {Author}, All Rights Reserved".format(
            Author=author, Year=year
        )


def resize_photos():
    def apply_result_info(result: Tuple[str, Dict[str, Any]]):
        key, info = result
        results[key] = info

    def error_callback(e: BaseException):
        logger.warning(f"photos: {e}")
        logger.debug("photos: An exception occurred", exc_info=e)

    debug = False
    resize_job_number: int = pelican_settings["PHOTO_RESIZE_JOBS"]

    if resize_job_number == -1:
        debug = True
        resize_job_number = 1
    elif resize_job_number == 0:
        resize_job_number = os.cpu_count() + 1

    logger.info(f"photos: Creating resize pool with {resize_job_number} worker(s) ...")

    pool = multiprocessing.Pool(processes=resize_job_number)
    logger.debug(f"Debug Status: {debug}")
    logger.info(f"photos: {len(DEFAULT_CONFIG['queue_resize'])} images in resize queue")
    results = {}
    for img_key, img in DEFAULT_CONFIG["queue_resize"].items():
        if debug:
            apply_result_info(img.process(img_key))
        else:
            pool.apply_async(
                img.process,
                args=(img_key,),
                callback=apply_result_info,
                error_callback=error_callback,
            )

    pool.close()
    pool.join()
    logger.info("photos: Applying results")
    for k, result_info in results.items():
        DEFAULT_CONFIG["queue_resize"][k].apply_result_info(result_info)


def detect_content(content):
    hrefs = None

    def replacer(m):
        what = m.group("what")
        value = m.group("value")
        tag = m.group("tag")
        output = m.group(0)

        if what not in ("photo", "lightbox"):
            # ToDo: Log unsupported type
            return output

        if value.startswith("/"):
            value = value[1:]

        if what == "photo":
            try:
                img = ContentImage(filename=value)
            except FileNotFound as e:
                logger.error(f"photos: {str(e)}")
                return output

            return "".join(
                (
                    "<",
                    m.group("tag"),
                    m.group("attrs_before"),
                    m.group("src"),
                    "=",
                    m.group("quote"),
                    os.path.join(pelican_settings["SITEURL"], img.image.web_filename),
                    m.group("quote"),
                    m.group("attrs_after"),
                )
            )

        elif what == "lightbox" and tag == "img":
            try:
                img = ContentImageLightbox(filename=value)
            except FileNotFound as e:
                logger.error(f"photos: {str(e)}")
                return output

            lightbox_attr_list = [""]

            gallery_name = value.split("/")[0]
            lightbox_attr_list.append(
                '{}="{}"'.format(
                    pelican_settings["PHOTO_LIGHTBOX_GALLERY_ATTR"], gallery_name
                )
            )

            if img.caption:
                lightbox_attr_list.append(
                    '{}="{}"'.format(
                        pelican_settings["PHOTO_LIGHTBOX_CAPTION_ATTR"],
                        str(img.caption),
                    )
                )

            lightbox_attrs = " ".join(lightbox_attr_list)

            return "".join(
                (
                    "<a href=",
                    m.group("quote"),
                    os.path.join(pelican_settings["SITEURL"], img.image.web_filename),
                    m.group("quote"),
                    lightbox_attrs,
                    "><img",
                    m.group("attrs_before"),
                    "src=",
                    m.group("quote"),
                    os.path.join(pelican_settings["SITEURL"], img.thumb.web_filename),
                    m.group("quote"),
                    m.group("attrs_after"),
                    "</a>",
                )
            )

        # else:
        #  logger.error("photos: No photo %s", value)

        return output

    if hrefs is None:
        regex = r"""
            <\s*
            (?P<tag>[^\s\>]+)  # detect the tag
            (?P<attrs_before>[^\>]*)
            (?P<src>href|src)  # match tag with src and href attr
            \s*=
            (?P<quote>["\'])  # require value to be quoted
            (?P<path>{}(?P<value>.*?))  # the url value
            (?P=quote)
            (?P<attrs_after>[^\>]*>)
        """.format(
            pelican_settings["INTRASITE_LINK_REGEX"]
        )
        hrefs = re.compile(regex, re.X)

    if content._content and (
        "{photo}" in content._content or "{lightbox}" in content._content
    ):
        content._content = hrefs.sub(replacer, content._content)


def galleries_string_decompose(gallery_string):
    splitter_regex = re.compile(r"[\s,]*?({photo}|{filename})")
    title_regex = re.compile(r"{(.+)}")
    galleries = map(
        str.strip,
        filter(None, splitter_regex.split(gallery_string)),
    )
    galleries = [
        gallery[1:] if gallery.startswith("/") else gallery for gallery in galleries
    ]
    if len(galleries) % 2 == 0 and " " not in galleries:
        galleries = zip(
            zip(["type"] * len(galleries[0::2]), galleries[0::2]),
            zip(["location"] * len(galleries[0::2]), galleries[1::2]),
        )
        galleries = [dict(gallery) for gallery in galleries]
        for gallery in galleries:
            title = re.search(title_regex, gallery["location"])
            if title:
                gallery["title"] = title.group(1)
                gallery["location"] = re.sub(
                    title_regex, "", gallery["location"]
                ).strip()
            else:
                gallery["title"] = DEFAULT_CONFIG["PHOTO_GALLERY_TITLE"]
        return galleries
    else:
        logger.error(
            f"Unexpected gallery location format! \n{pprint.pformat(galleries)}"
        )


def process_content_galleries(content: Union[Article, Page], location):
    """
    Process all galleries attached to an article or page.

    :param content: The content object
    :param location: Galleries
    """
    photo_galleries = []

    galleries = galleries_string_decompose(location)

    for gallery in galleries:
        try:
            gallery = Gallery(content, gallery)
        except GalleryNotFound as e:
            logger.error(f"photos: {str(e)}")

        photo_galleries.append(gallery)
    return photo_galleries


def detect_content_galleries(
    generator: Union[ArticlesGenerator, PagesGenerator], content: Union[Article, Page]
):
    def replace_gallery_string(pattern_match):
        photo_galleries = process_content_galleries(
            content, pattern_match.group("gallery_name")
        )
        template = generator.get_template(
            pelican_settings["PHOTO_INLINE_GALLERY_TEMPLATE"]
        )
        template_values = {
            "galleries": photo_galleries,
        }
        if isinstance(generator, ArticlesGenerator):
            template_values["article"] = content
        elif isinstance(generator, PagesGenerator):
            template_values["page"] = content
        return template.render(**template_values)

    # print(content.content)
    if "gallery" in content.metadata:
        gallery = content.metadata.get("gallery")
        if gallery.startswith("{photo}") or gallery.startswith("{filename}"):
            content.photo_gallery = process_content_galleries(content, gallery)
        elif gallery:
            logger.error(f"photos: Gallery tag not recognized: {gallery}")

    if pelican_settings["PHOTO_INLINE_GALLERY_ENABLED"]:
        content._content = re.sub(
            pelican_settings["PHOTO_INLINE_GALLERY_PATTERN"],
            replace_gallery_string,
            content._content,
        )


def image_clipper(x):
    return x[8:] if x[8] == "/" else x[7:]


def file_clipper(x):
    return x[11:] if x[10] == "/" else x[10:]


def prepare_config(generator: pelican.generators.Generator):
    for name in ("PHOTO_ARTICLE", "PHOTO_GALLERY", "PHOTO_THUMB"):
        if isinstance(pelican_settings[name], (list, tuple)):
            logger.info(f"Converting legacy config to new values: {name}")
            pelican_settings[name] = {
                "default": {
                    "width": pelican_settings[name][0],
                    "height": pelican_settings[name][1],
                    "type": "jpeg",
                    "options": {"quality": pelican_settings[name][2]},
                }
            }


def process_image(generator, content, image):
    try:
        content.photo_image = ArticleImage(
            content=content, filename=image, generator=generator
        )
    except FileNotFound as e:
        logger.error(f"photo: {str(e)}")


def detect_image(generator, content):
    image = content.metadata.get("image", None)
    if image:
        if image.startswith("{photo}") or image.startswith("{filename}"):
            process_image(generator, content, image)
        else:
            logger.error(f"photos: Image tag not recognized: {image}")


def detect_images_and_galleries(generators):
    """Runs generator on both pages and articles."""
    for generator in generators:
        if isinstance(generator, ArticlesGenerator):
            article: Article
            for article in itertools.chain(
                generator.articles, generator.translations, generator.drafts
            ):
                detect_image(generator, article)
                detect_content_galleries(generator, article)
        elif isinstance(generator, PagesGenerator):
            page: Page
            for page in itertools.chain(
                generator.pages, generator.translations, generator.hidden_pages
            ):
                detect_image(generator, page)
                detect_content_galleries(generator, page)


def handle_signal_all_generators_finalized(generators: pelican.generators.Generator):
    detect_images_and_galleries(generators)
    resize_photos()


def register():
    """Uses the new style of registration based on GitHub Pelican issue #314."""
    signals.initialized.connect(initialized)
    try:
        signals.generator_init.connect(prepare_config)
        signals.content_object_init.connect(detect_content)
        signals.all_generators_finalized.connect(handle_signal_all_generators_finalized)
    except Exception as e:
        logger.exception(f"Plugin failed to execute: {pprint.pformat(e)}")
