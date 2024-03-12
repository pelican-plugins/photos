import abc
import base64
from collections import namedtuple
import datetime
from functools import wraps
from html.parser import HTMLParser
import itertools
import json
import logging
import mimetypes
import multiprocessing
import multiprocessing.pool
import os
import pprint
import re
import time
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, Union
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
pelican_photo_inline_galleries = {}
g_generator = None
g_image_queue = []
g_profiles = {}
g_profiling_call_level = 0
g_process_pool: Optional[multiprocessing.Pool] = None
g_process_pool_initialized: bool = False
g_image_cache: Dict[str, "Image"] = {}

try:
    from PIL import (
        ExifTags,
        Image as PILImage,
        ImageDraw,
        ImageEnhance,
        ImageFont,
        ImageOps,
    )
except ImportError:
    logger.error("PIL/Pillow not found")  # noqa: TRY400
    raise

EXIF_TAGS_NAME_CODE = {v: n for n, v in ExifTags.TAGS.items()}

try:
    import piexif
except ImportError:
    ispiexif = False
    logger.warning("piexif not found! Cannot use exif manipulation features")
else:
    ispiexif = True
    logger.debug("piexif found.")


InlineContentData = namedtuple(
    "InlineContentData", ["type", "image", "html_attributes"]
)


class BaseNoteParseError(Exception):
    pass


class InternalError(Exception):
    pass


class FileNotFound(Exception):
    pass


class GalleryNotFound(Exception):
    pass


class FileExcluded(Exception):
    pass


class ImageExcluded(FileExcluded):
    pass


class ImageConfigNotFound(Exception):
    pass


class ProfileNotFound(Exception):
    pass


class HTMLTagParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tag_attrs = {}

    def handle_starttag(self, tag, attrs):
        logger.debug(f"Found tag: {tag}")
        self.tag_attrs.update(dict(attrs))


class HTMLImgParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.img_attrs = ()

    def handle_starttag(self, tag, attrs):
        logger.debug(f"Found tag: {tag}")
        if tag == "img":
            self.img_attrs = attrs


class Profile:
    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        default_profile: Optional["Profile"] = None,
    ):
        self.name = name
        self._config = config
        self._default_profile = default_profile

    def get_image_config(self, name) -> Dict[str, Any]:
        images = self._config.get("images")
        config = None
        if images is not None:
            config = images.get(name)
        if config is None and self._default_profile is not None:
            config = self._default_profile.get_image_config(name)
        if config is None:
            raise ImageConfigNotFound(  # noqa: TRY003
                f"Unable to find image config for '{name}' in profile '{self.name}'"
            )
        return config

    def render_template(self, default_template_name=None, **kwargs):
        tpl_name = self._config.get("template_name")
        if tpl_name is None:
            tpl_name = default_template_name
        if tpl_name is None:
            logger.error("Unable to find template for profile")

        context_vars = {"SITEURL": pelican_settings["SITEURL"]}
        context_vars.update(kwargs)
        return g_generator.get_template(tpl_name).render(**context_vars)

    @property
    def file_suffix(self) -> str:
        return self._config.get("file_suffix", "")

    @property
    def has_template(self) -> bool:
        if "template_name" in self._config:
            return True
        return False

    @property
    def article_file_suffix(self) -> str:
        return self.get_image_config("article").get("file_suffix", "a")

    @property
    def article_html_img_attributes(self) -> Dict[str, str]:
        return self.get_image_config("article").get("html_img_attributes", {})

    @property
    def article_image_spec(self) -> Dict[str, Any]:
        return self.get_image_config("article")["specs"]

    @property
    def gallery_file_suffix(self) -> str:
        return self.get_image_config("gallery").get("file_suffix", "")

    @property
    def gallery_html_img_attributes(self) -> Dict[str, str]:
        return self.get_image_config("gallery").get("html_img_attributes", {})

    @property
    def gallery_image_spec(self) -> Dict[str, Any]:
        return self.get_image_config("gallery")["specs"]

    @property
    def thumb_file_suffix(self) -> str:
        return self.get_image_config("thumb").get("file_suffix", "t")

    @property
    def thumb_html_img_attributes(self) -> Dict[str, str]:
        return self.get_image_config("thumb").get("html_img_attributes", {})

    @property
    def thumb_image_spec(self) -> Dict[str, Any]:
        return self.get_image_config("thumb")["specs"]


def measure_time(func):
    @wraps(func)
    def measure_time_wrapper(*args, **kwargs):
        if pelican_settings.get("PHOTO_PROFILING_ENABLED") is not True:
            return func(*args, **kwargs)

        global g_profiling_call_level

        msg_prefix_start = ""
        msg_prefix_end = ""
        if g_process_pool is None and g_profiling_call_level > 0:
            msg_prefix_start = "|" * g_profiling_call_level + "-> "
            msg_prefix_end = "|" * (g_profiling_call_level - 1) + "'-> "

        logger.debug(
            f"{msg_prefix_start}Calling {func.__name__}()"
            f" with args {args} and kwargs {kwargs}"
        )
        start_time = time.perf_counter()
        # resize_job_number == -1 -> no multiprocessing
        if g_process_pool is None:
            g_profiling_call_level += 1
        result = func(*args, **kwargs)
        if g_process_pool is None:
            g_profiling_call_level -= 1
        end_time = time.perf_counter()
        total_time = end_time - start_time
        logger.debug(
            f"{msg_prefix_end}Call {func.__name__}() took {total_time:.4f} seconds"
        )
        return result

    return measure_time_wrapper


@measure_time
def find_profile(names: List[str], default_not_found=True):
    """Find first matching profile."""
    for name in names:
        try:
            return get_profile(name)
        except ProfileNotFound:
            pass
    if default_not_found:
        return get_profile("default")

    raise ProfileNotFound(f"Unable to find any of the profiles: {', '.join(names)}")


def get_profile(name: str) -> Profile:
    """Return the profile."""
    profile = g_profiles.get(name)
    if profile is None:
        raise ProfileNotFound(f"Unable to find profile '{name}'")
    return profile


def get_image_from_string(
    url_string: str,
    image_class: Optional[Type["BaseImage"]] = None,
    default_image_class: Optional[Type["BaseImage"]] = None,
) -> "BaseImage":
    value_mapping = {
        "profile": "profile_name",
    }
    url = urllib.parse.urlparse(url_string)
    query_params = dict(urllib.parse.parse_qsl(url.query))
    kwargs = {}
    for query_name, arg_name in value_mapping.items():
        if query_name in query_params:
            kwargs[arg_name] = query_params.get(query_name)

    if image_class is None and "type" in query_params:
        image_type = query_params["type"].lower().strip()
        if image_type == "content":
            image_class = ContentImage
        else:
            logger.warning(
                f"Unable to find image class for type='{image_type}'"
                f" on '{url_string}' - Supported types are: content"
            )

    if image_class is None and default_image_class is not None:
        logger.debug(f"Using default image class for '{url_string}'")
        image_class = default_image_class

    return image_class(filename=url.path, **kwargs)


class BaseNoteCache(abc.ABC):
    note_cache: ClassVar[Dict[str, "BaseNoteCache"]] = {}

    @abc.abstractclassmethod
    def note_filename(cls):
        raise NotImplementedError

    def __init__(self, filename):
        self.filename = filename
        self.note_cache[self.filename] = self
        self.notes: Dict[str, str] = {}
        self._read()

    @abc.abstractmethod
    def _parse_line(self, line: str) -> Tuple[str, Any]:
        raise NotImplementedError("Line parser not implemented")

    def _read(self):
        if not os.path.isfile(self.filename):
            return

        try:
            logger.debug(f"Reading information from {self.filename}")
            with pelican_open(self.filename) as text:
                for line_number, line in enumerate(text.splitlines()):
                    line = line.strip()  # noqa: PLW2901

                    # skip empty lines
                    if len(line) == 0:
                        continue

                    # skip comments
                    if line.startswith("#"):
                        continue

                    try:
                        result = self._parse_line(line)
                        self.notes[result[0]] = result[1]
                    except BaseNoteParseError as e:
                        logger.warning(
                            f"Parser error in file {self.filename}"
                            f" on line {line_number + 1}: {e}"
                        )

        except Exception:
            logger.exception(
                f"There was an error while processing the {self.filename}. "
            )

    def get_value(self, source_image):
        return self.notes.get(os.path.basename(source_image.filename))

    @classmethod
    def from_cache(cls, source_image: "SourceImage"):
        filename = os.path.join(
            os.path.dirname(source_image.filename), cls.note_filename()
        )
        notes = cls.note_cache.get(filename)
        if notes is None:
            notes = cls(filename=filename)
        return notes


class BaseNoteKeyCache(BaseNoteCache):
    def _parse_line(self, line: str) -> Tuple[str, bool]:
        return line.strip(), True


class BaseNoteKeyValueCache(BaseNoteCache):
    def _parse_line(self, line: str) -> Tuple[str, str]:
        # parse content
        m = line.split(":", 1)
        if len(m) > 1:
            return m[0].strip(), m[1].strip()

        raise BaseNoteParseError(
            "Wrong format. Expecting '<filename>: <value>' per line"
        )


class CaptionCache(BaseNoteKeyValueCache):
    @classmethod
    def note_filename(cls):
        return "captions.txt"


class ExifCache(BaseNoteKeyValueCache):
    @classmethod
    def note_filename(cls):
        return "exif.txt"


class ExcludeCache(BaseNoteKeyCache):
    @classmethod
    def note_filename(cls):
        return "blacklist.txt"


class BaseNote:
    cache_class = None

    def __init__(self, source_image):
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


class BaseImage:
    regex_filename = re.compile(r"^({(?P<type>photo|filename)})?(?P<filename>.*)")

    def __init__(
        self,
        filename: str,
        profile_name: Optional[str] = None,
        profile: Optional[Profile] = None,
    ):
        m = self.regex_filename.match(filename)

        self.filename_raw = filename

        self.filename: str = m.group("filename")
        self.filename_type: Optional[str] = m.group("type")

        self.src_filename = os.path.join(
            os.path.expanduser(pelican_settings["PHOTO_LIBRARY"]), self.filename
        )

        self.file = os.path.basename(self.filename).lower()

        self.profile = profile

        if self.profile is None:
            if profile_name is None:
                profile_name = "default"
            try:
                self.profile = get_profile(profile_name)
            except ProfileNotFound:
                logger.error(
                    f"Unable to find profile '{profile_name}' for image '{filename}'"
                )
                raise


class ArticleImage(BaseImage):
    """Images/photos on the top of an article or page.

    :param content: Internal content object
    :param filename: The filename of the image
    :param generator: The generator
    """

    def __init__(
        self,
        content: pelican.contents.Content,
        filename: str,
        profile_name: Optional[str] = None,
        profile: Optional[Profile] = None,
    ):
        super().__init__(filename, profile_name=profile_name, profile=profile)

        if self.filename_type == "filename":
            self.src_filename = os.path.join(
                g_generator.path, content.relative_dir, self.filename
            )
        elif self.filename_type != "photo":
            raise InternalError(f"Unable to detect image type {self.filename_raw}")

        if not os.path.isfile(self.src_filename):
            raise FileNotFound(
                f"No photo for {content.source_path} at {self.filename} "
                f"source {self.src_filename}"
            )

        photo = (
            os.path.splitext(self.filename)[0].lower()
            + self.profile.article_file_suffix
            + self.profile.file_suffix
        )
        thumb = (
            os.path.splitext(self.filename)[0].lower()
            + self.profile.thumb_file_suffix
            + self.profile.file_suffix
        )
        img = Image(
            src=self.src_filename,
            dst=os.path.join("photos", photo),
            specs=self.profile.article_image_spec,
        )
        self.image = enqueue_image(img)
        img = Image(
            src=self.src_filename,
            dst=os.path.join("photos", thumb),
            specs=self.profile.thumb_image_spec,
            is_thumb=True,
        )
        self.thumb = enqueue_image(img)

    def __getitem__(self, index):
        """Legacy support."""
        if index == 0:
            return self.file

        if index == 1:
            return self.image.web_filename

        if index == 2:
            return self.thumb.web_filename

        raise IndexError


class ContentImage(BaseImage):
    def __init__(
        self,
        filename,
        profile_name: Optional[str] = None,
        profile: Optional[Profile] = None,
    ):
        super().__init__(filename, profile_name=profile_name, profile=profile)

        if not os.path.isfile(self.src_filename):
            raise FileNotFound(f"No photo for {self.src_filename} {self.filename}")

        img = Image(
            src=self.src_filename,
            dst=os.path.join(
                "photos",
                os.path.splitext(self.filename)[0].lower()
                + self.profile.article_file_suffix
                + self.profile.file_suffix,
            ),
            specs=self.profile.article_image_spec,
        )
        self.image = enqueue_image(img)

    @property
    def caption(self) -> Optional[Caption]:
        return self.image.caption


class ContentImageLightbox(BaseImage):
    def __init__(
        self,
        filename,
        profile_name: Optional[str] = None,
        profile: Optional[Profile] = None,
    ):
        super().__init__(filename, profile_name=profile_name, profile=profile)

        if not os.path.isfile(self.src_filename):
            raise FileNotFound(f"No photo for {self.src_filename} l {self.filename}")

        img = Image(
            src=self.src_filename,
            dst=os.path.join(
                "photos",
                os.path.splitext(filename)[0].lower()
                + self.profile.gallery_file_suffix
                + self.profile.file_suffix,
            ),
            specs=self.profile.gallery_image_spec,
        )
        self.image = enqueue_image(img)

        img = Image(
            src=self.src_filename,
            dst=os.path.join(
                "photos",
                os.path.splitext(filename)[0].lower()
                + self.profile.thumb_file_suffix
                + self.profile.file_suffix,
            ),
            specs=self.profile.thumb_image_spec,
            is_thumb=True,
        )
        self.thumb = enqueue_image(img)

    @property
    def caption(self) -> Optional[Caption]:
        return self.image.caption


class Gallery:
    """Process a single gallery.

    - look for images
    - read meta data
    - read exif data
    - enqueue the images to be processed
    """

    def __init__(  # noqa: PLR0912 -- too many branches
        self,
        content: Union[Article, Page],
        location_parsed,
        profile_name: Optional[str] = None,
        profile: Optional[Profile] = None,
    ):
        self.content = content

        if profile is None:
            if profile_name is None:
                profile_name = "default"
            self.profile = get_profile(profile_name)
        else:
            self.profile = profile

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

        image_filenames = []
        logger.info(f"photos: Gallery detected: {rel_gallery}")
        for pic in sorted(os.listdir(dir_gallery)):
            if pic.startswith("."):
                continue
            if pic.endswith(".txt"):
                continue
            if not os.path.isfile(os.path.join(dir_gallery, pic)):
                continue
            image_filename = os.path.join(location_parsed["location"], pic)
            image_filenames.append(f"{location_parsed['type']}{image_filename}")

        self.dst_dir = os.path.join("photos", rel_gallery.lower())
        self.images: List[GalleryImage] = []

        self.title = location_parsed["title"]
        for pic in image_filenames:
            try:
                self.images.append(
                    GalleryImage(
                        filename=pic,
                        gallery=self,
                        profile=self.profile,
                    )
                )
            except ImageExcluded:
                logger.debug(f"photos: Image {pic} excluded")
            except FileExcluded as e:
                logger.debug(f"photos: File {pic} excluded: {e!s}")

    def __getitem__(self, item):
        if item == 0:
            return self.title

        if item == 1:
            return self.images

        raise IndexError


class GalleryImage(BaseImage):
    """Image of a gallery."""

    def __init__(
        self,
        filename,
        gallery: Gallery,
        profile_name: Optional[str] = None,
        profile: Optional[Profile] = None,
    ):
        super().__init__(filename, profile_name=profile_name, profile=profile)

        #: The gallery this image belongs to
        self._gallery = gallery

        if self.filename_type == "filename":
            self.src_filename = os.path.join(
                g_generator.path, self._gallery.content.relative_dir, self.filename
            )

        img = Image(
            src=self.src_filename,
            dst=os.path.join(
                "photos",
                os.path.splitext(self.filename)[0].lower()
                + self.profile.gallery_file_suffix
                + self.profile.file_suffix,
            ),
            specs=self.profile.gallery_image_spec,
        )
        if img.is_excluded:
            raise ImageExcluded("Image excluded from gallery")

        #: The image object
        self.image = enqueue_image(img)

        img = Image(
            src=self.src_filename,
            dst=os.path.join(
                "photos",
                os.path.splitext(self.filename)[0].lower()
                + self.profile.thumb_file_suffix
                + self.profile.file_suffix,
            ),
            specs=self.profile.thumb_image_spec,
            is_thumb=True,
        )
        #: The thumbnail
        self.thumb = enqueue_image(img)

    def __getitem__(self, item):
        """Legacy support."""
        if item == 0:
            return self.file

        if item == 1:
            return self.image

        if item == 2:
            return self.thumb

        if item == 3:
            return "" if self.exif is None else self.exif.value

        if item == 4:
            return "" if self.caption is None else self.caption.value

        raise IndexError

    @property
    def caption(self) -> Optional[Caption]:
        return self.image.caption

    @property
    def exif(self) -> Optional[Exif]:
        return self.image.exif

    @property
    def is_excluded(self) -> bool:
        """Is the image is excluded from the gallery."""
        return self.image.is_excluded


class GlobalImage(BaseImage):
    def __init__(
        self,
        filename,
        profile_name: Optional[str] = None,
        profile: Optional[Profile] = None,
    ):
        super().__init__(filename, profile_name=profile_name, profile=profile)

        if not os.path.isfile(self.src_filename):
            raise FileNotFound(f"No photo for {self.src_filename} {self.filename}")

        img = Image(
            src=self.src_filename,
            dst=os.path.join(
                "photos",
                os.path.splitext(self.filename)[0].lower()
                + self.profile.article_file_suffix
                + self.profile.file_suffix,
            ),
            specs=self.profile.article_image_spec,
        )
        self.image = enqueue_image(img)


class Image:
    """The main Image class.

    This class holds all information of the generated image that is required to process
    the image.

    """

    def __init__(  # noqa: PLR0912, PLR0913, PLR0915 -- this function does a _lot_
        self,
        src,
        dst,
        spec: Optional[Dict[str, Any]] = None,
        specs: Optional[Dict[str, Dict[str, Any]]] = None,
        is_thumb=False,
    ):
        if spec is not None and specs is not None:
            raise ValueError("Both spec and specs must not be provided")

        #: The source image
        self.source_image = SourceImage.from_cache(src)
        self.dst = dst
        self.is_thumb = is_thumb

        self._average_color = None
        self._height: Optional[int] = None
        self._width: Optional[int] = None
        self._result_info_loaded = False
        self._result_info_allowed_names = ("_average_color", "_height", "_width")
        self.images = {}

        #: The exif data used for the result image
        self.exif_result: Optional[dict] = None

        #: The exif data from the source image
        self.exif_orig: Optional[dict] = None

        #: The icc profile from the source image
        self.icc_profile: Optional[bytes] = None

        # We create new instances from this class in the main process
        # so we can access the global pelican_settings. We attach them
        # as class var to push them to the processes when running in
        # spawn mode.
        #: Global pelican settings
        self._pelican_settings = pelican_settings

        if spec is None:
            if specs is None:
                raise ValueError("Only one of spec and specs must be provided")
            spec = specs.get(self.source_image.type)
            if spec is None:
                spec = specs["default"]

        #: The specification how to transform the source image
        self.spec: Dict[str, Any] = spec.copy()

        #: Image type e.g. jpeg, webp
        self.type = spec["type"].lower()

        image_options: Dict[str, Any] = self._pelican_settings[
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

        srcset_specs: Optional[list, tuple] = self.spec.get("srcset")
        if not isinstance(srcset_specs, (list, tuple)):
            srcset_specs = []

        #: The srcset for the image used in HTML
        self.srcset = ImageSrcSet()
        for srcset_spec in srcset_specs:
            img = SrcSetImage(
                src=self.source_image.filename,
                dst=self.dst,
                spec=srcset_spec,
                is_thumb=self.is_thumb,
            )
            self.srcset.append(enqueue_image(img))

        additional_images: Dict[str, Any] = spec.get("images")
        if additional_images is None:
            additional_images = {}

        for add_img_name, add_img_spec in additional_images.items():
            if not isinstance(add_img_spec, dict):
                add_img_spec = {}  # noqa: PLW2901

            add_img_spec_combined = {}
            # We use some values from the parent as default
            for name in ("operations", "skip_operations", "type"):
                if name in self.spec:
                    add_img_spec_combined[name] = self.spec[name]
            add_img_spec_combined.update(add_img_spec)

            img = Image(
                src=self.source_image.filename,
                dst=f"{self.dst}_{add_img_name}",
                spec=add_img_spec_combined,
                is_thumb=self.is_thumb,
            )
            img = enqueue_image(img)
            self.images[add_img_name] = img

        #: The name of the output file
        self.output_filename = "{filename}.{extension}".format(
            filename=os.path.join(pelican_output_path, self.dst),
            extension=self._pelican_settings["PHOTO_FILE_EXTENSIONS"].get(
                self.type, self.type
            ),
        )

        #: The name and path for the web page
        self.web_filename = "{resized}.{extension}".format(
            resized=self.dst,
            extension=self._pelican_settings["PHOTO_FILE_EXTENSIONS"].get(
                self.type, self.type
            ),
        )

        self.pre_operations: List[Union[str, list, tuple]] = []
        self.operations: List[Union[str, list, tuple]] = []
        self.post_operations: List[Union[str, list, tuple]] = []

        self.pre_operations.append("exif.rotate")
        self.pre_operations.append("exif.manipulate")

        if self.type in ("jpeg",):
            self.pre_operations.append("main.remove_alpha")

        self.post_operations.append("main.watermark")

        if self.type == "gif":
            self.post_operations.append("main.convert_mode_p")

        if self.type == "jpeg":
            self.post_operations.append("main.convert_mode_rgb")

        skip_operations = self.spec.get("skip_operations")
        if isinstance(skip_operations, (list, tuple)):
            for item in self.pre_operations:
                if (isinstance(item, str) and item in skip_operations) or (
                    isinstance(item, (list, tuple)) and item[0] in skip_operations
                ):
                    self.pre_operations.remove(item)
            for item in self.post_operations:
                if (isinstance(item, str) and item in skip_operations) or (
                    isinstance(item, (list, tuple)) and item[0] in skip_operations
                ):
                    self.post_operations.remove(item)

        operations = self.spec.get("operations")
        if operations is None:
            if self.is_thumb and self._pelican_settings["PHOTO_SQUARE_THUMB"]:
                self.operations.append("ops.fit")
            else:
                self.operations.append("main.resize")
        elif isinstance(operations, (list, tuple)):
            self.operations = operations
        else:
            logger.warning("Wrong data-type for operations, should be list or tuple")
            self.operations = []

        operations = []
        for operation in self.operations:
            if isinstance(operation, str):
                operation_name = operation
                operation_args = {}
            else:
                operation_name = operation[0]
                operation_args = operation[1]

            if operation_name == "main.resize" and not operation_args:
                operation_args["resample"] = PILImage.LANCZOS
            if operation_name == "ops.fit" and not operation_args:
                operation_args["method"] = PILImage.LANCZOS
            if (
                operation_name in ("main.resize", "ops.fit")
                and "size" not in operation_args
            ):
                operation_args["size"] = (spec["width"], spec["height"])

            operations.append((operation_name, operation_args))

        self.operations = operations

        self.operation_mappings = {
            "main.convert": self._operation_convert,
            "main.convert_mode_p": self._operation_convert_mode_p,
            "main.convert_mode_rgb": self._operation_convert_mode_rgb,
            "main.remove_alpha": self._operation_remove_alpha,
            "main.resize": self._operation_resize,
            "main.quantize": self._operation_quantize,
            "main.watermark": self._operation_watermark,
            "ops.greyscale": ImageOps.grayscale,
            "ops.fit": ImageOps.fit,
        }
        self.advanced_operation_mappings = {
            "exif.rotate": self._operation_exif_rotate,
            "exif.manipulate": self._operation_manipulate_exif,
        }

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(source_image={self.source_image}, "
            f"dst={self.dst} is_thumb={self.is_thumb})"
        )

    def __str__(self):
        return self.web_filename

    @property
    def average_color(self) -> Optional[str]:
        """Average color."""
        if self._average_color is None:
            self._load_result_info()
        if self._average_color:
            return "#{:02x}{:02x}{:02x}".format(*self._average_color)

        return None

    @property
    def caption(self) -> Optional[Caption]:
        """Caption of the image."""
        return self.source_image.caption

    @property
    def data(self) -> bytes:
        with open(self.output_filename, "rb") as fp:
            return fp.read()

    @property
    def data_base64(self) -> str:
        return base64.b64encode(self.data).decode("UTF-8")

    @property
    def exif(self) -> Optional[Exif]:
        return self.source_image.exif

    @property
    def is_excluded(self) -> bool:
        """Is the image on the exclude-list."""
        return self.source_image.is_excluded

    @staticmethod
    def is_alpha(img: PILImage.Image) -> bool:
        return bool(
            img.mode in ("RGBA", "LA") or img.mode == "P" and "transparency" in img.info
        )

    @property
    def height(self) -> int:
        """Height of the image."""
        if self._height is None:
            self._load_result_info()
        return self._height

    @property
    def width(self) -> int:
        """Width of the image."""
        if self._width is None:
            self._load_result_info()
        return self._width

    def _load_result_info(self, image: Optional[PILImage.Image] = None):
        """Load the information from the result image."""
        if not self._result_info_loaded:
            if image is None:
                image: PILImage.Image = PILImage.open(self.output_filename)

            if self._pelican_settings["PHOTO_RESULT_IMAGE_AVERAGE_COLOR"]:
                image2: PILImage.Image = image.resize((1, 1), PILImage.LANCZOS)
                # We need RGB to get red, green and blue values for the pixel
                self._average_color = image2.convert("RGB").getpixel((0, 0))

            self._height = image.height
            self._width = image.width

        results = {}
        for name in self._result_info_allowed_names:
            results[name] = getattr(self, name)
        self._result_info_loaded = True
        return results

    def apply_result_info(self, info: Dict[str, Any]):
        """Apply the information from the result image.

        This occurs if it has been processed in a different process.
        """
        for name in self._result_info_allowed_names:
            if name in info:
                setattr(self, name, info[name])
        self._result_info_loaded = True

    def process(  # noqa: PLR0912, PLR0915 -- this function does a lot
        self,
    ) -> Tuple[str, Dict[str, Any]]:
        """Process the image."""
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
            return self.dst, self._load_result_info()

        image = self.source_image.open()
        if "icc_profile" in image.info:
            self.icc_profile = image.info["icc_profile"]

        self.exif_orig = image.getexif()
        if ispiexif:
            if self._pelican_settings["PHOTO_EXIF_KEEP"] and "exif" in image.info:
                # Copy the exif data if we want to keep it
                try:
                    self.exif_result = piexif.load(image.info["exif"])
                except Exception as e:  # noqa: BLE001 -- we are just warning here
                    logger.warning(
                        "There was an error reading exif data from"
                        f" '{self.source_image.filename}': {e}"
                    )
            else:
                self.exif_result = {}
        else:
            logger.info("Unable to keep exif data if piexif is not installed")

        operations = self.pre_operations + self.operations + self.post_operations
        for i, operation in enumerate(operations):
            operation_args = []
            operation_kwargs = {}
            if isinstance(operation, str):
                operation_name = operation
            else:
                operation_name = operation[0]
                if len(operation) > 1:
                    if isinstance(operation[1], (list, tuple)):
                        operation_args = operation[1]
                    elif isinstance(operation[1], (dict,)):
                        operation_kwargs = operation[1]
                if len(operation) > 2 and operation_name[2] is not None:
                    operation_kwargs = operation[2]

            logger.debug(f"Processing({self.output_filename}): {i}={operation_name}")
            func = self.operation_mappings.get(operation_name)
            advanced_func = self.advanced_operation_mappings.get(operation_name)
            if func:
                image = func(image, *operation_args, **operation_kwargs)
            elif advanced_func:
                image = advanced_func(image, self, *operation_args, **operation_kwargs)
            else:
                logger.warning(
                    f"Unable to find operation: '{operation_name}' "
                    f"for destination image '{self.dst}'"
                )

        image_options = self.spec.get("options", {})
        directory = os.path.dirname(self.output_filename)
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
            except Exception:
                logger.exception(f"Could not create {directory}")

        exif_data = b""
        if ispiexif and self.exif_result:
            try:
                # Prevent error with piexif issue
                # "dump" got wrong type of exif value. 41729 in Exif IFD.
                # Got as <class 'int'>.
                # See bug https://github.com/hMatoba/Piexif/issues/95
                self.exif_result.pop(piexif.ExifIFD.SceneType, None)

                exif_data = piexif.dump(self.exif_result)
            except Exception as e:  # noqa: BLE001 -- we are just warning here
                logger.warning(
                    "There was an error dumping exif data for image "
                    f" '{self.output_filename}': {e}"
                )
                exif_data = b""

        image.save(
            self.output_filename,
            self.type,
            icc_profile=self.icc_profile,
            exif=exif_data,
            **image_options,
        )
        return self.dst, self._load_result_info(image=image)

    def reduce_opacity(self, im: PILImage.Image, opacity) -> PILImage.Image:
        """Reduces Opacity.

        Returns an image with reduced opacity.
        Taken from http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/362879
        """
        assert opacity >= 0 and opacity <= 1
        im = im.copy() if self.is_alpha(im) else im.convert("RGBA")

        alpha = im.split()[3]
        alpha = ImageEnhance.Brightness(alpha).enhance(opacity)
        im.putalpha(alpha)
        return im

    @staticmethod
    def _operation_convert(image: PILImage.Image, *args, **kwargs):
        return image.convert(*args, **kwargs)

    @staticmethod
    def _operation_convert_mode_p(img: PILImage.Image) -> PILImage.Image:
        """Convert image into P mode if not already in this mode."""
        if img.mode == "P":
            return img
        return img.convert("P")

    @staticmethod
    def _operation_convert_mode_rgb(img: PILImage.Image) -> PILImage.Image:
        """Convert image into RGB mode if not already in this mode."""
        if img.mode == "RGB":
            return img
        return img.convert("RGB")

    @staticmethod
    def _operation_exif_rotate(
        image: PILImage.Image, image_meta: "Image"
    ) -> PILImage.Image:
        """Rotate the image with the information from exif data."""
        orientation = image_meta.exif_orig.get(EXIF_TAGS_NAME_CODE["Orientation"])
        if orientation is None:
            return image
        if orientation == 2:
            image = image.transpose(PILImage.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            image = image.rotate(180)
        elif orientation == 4:
            image = image.rotate(180).transpose(PILImage.FLIP_LEFT_RIGHT)
        elif orientation == 5:
            image = image.rotate(-90).transpose(PILImage.FLIP_LEFT_RIGHT)
        elif orientation == 6:
            image = image.rotate(-90, expand=True)
        elif orientation == 7:
            image = image.rotate(90).transpose(PILImage.FLIP_LEFT_RIGHT)
        elif orientation == 8:
            image = image.rotate(90)

        if ispiexif and image_meta.exif_result:
            image_meta.exif_result.setdefault("0th", {})
            image_meta.exif_result["0th"][piexif.ImageIFD.Orientation] = 1

        return image

    def _operation_manipulate_exif(
        self, image: PILImage.Image, image_meta: "Image"
    ) -> PILImage.Image:
        if image_meta.exif_result is None:
            return image

        if self._pelican_settings["PHOTO_EXIF_REMOVE_GPS"]:
            # Remove GPS data from exif
            # but don't raise an exception if it does not exist
            image_meta.exif_result.pop("GPS", None)

        if self._pelican_settings["PHOTO_EXIF_COPYRIGHT"]:
            image_meta.exif_result.setdefault("0th", {})
            # Be minimally destructive to any preset EXIF author or copyright
            # information. If there is copyright or author information, prefer that
            # over everything else.
            author = self._pelican_settings["PHOTO_EXIF_COPYRIGHT_AUTHOR"]

            if not image_meta.exif_result["0th"].get(piexif.ImageIFD.Artist):
                image_meta.exif_result["0th"][piexif.ImageIFD.Artist] = author

            if not image_meta.exif_result["0th"].get(piexif.ImageIFD.Copyright):
                license = build_license(
                    self._pelican_settings["PHOTO_EXIF_COPYRIGHT"], author
                )
                image_meta.exif_result["0th"][piexif.ImageIFD.Copyright] = license

        return image

    def _operation_remove_alpha(self, image: PILImage.Image) -> PILImage.Image:
        """Remove the alpha channel."""
        if not self.is_alpha(image):
            return image
        background = PILImage.new(
            "RGB", image.size, self._pelican_settings["PHOTO_ALPHA_BACKGROUND_COLOR"]
        )
        background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
        return background

    @staticmethod
    def _operation_resize(image, *args, **kwargs):
        image.thumbnail(*args, **kwargs)
        return image

    @staticmethod
    def _operation_quantize(image: PILImage.Image, *args, **kwargs):
        return image.quantize(*args, **kwargs)

    def _operation_watermark(self, image: PILImage.Image) -> PILImage.Image:
        """Add the watermark."""
        if not self._pelican_settings["PHOTO_WATERMARK"]:
            return image
        if self.is_thumb and not self._pelican_settings["PHOTO_WATERMARK_THUMB"]:
            return image
        margin = [10, 10]
        opacity = 0.6

        watermark_layer = PILImage.new("RGBA", image.size, (0, 0, 0, 0))
        draw_watermark = ImageDraw.Draw(watermark_layer)
        text_reducer = 32
        image_reducer = 8
        text_size = [0, 0]
        mark_size = [0, 0]

        if self._pelican_settings["PHOTO_WATERMARK_TEXT"]:
            font_name = "SourceCodePro-Bold.otf"
            default_font = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), font_name
            )
            font_size = watermark_layer.size[0] // text_reducer
            if font_size < 1:
                logger.info(
                    f"photos: font size {font_size} is to small skipping watermark"
                )
                return image

            font = ImageFont.FreeTypeFont(font=default_font, size=font_size)
            text_size[0] = draw_watermark.textlength(
                self._pelican_settings["PHOTO_WATERMARK_TEXT"], font, direction="rtl"
            )
            text_size[1] = font.size
            text_position = [image.size[i] - text_size[i] - margin[i] for i in [0, 1]]
            draw_watermark.text(
                text_position,
                self._pelican_settings["PHOTO_WATERMARK_TEXT"],
                self._pelican_settings["PHOTO_WATERMARK_TEXT_COLOR"],
                font=font,
            )

        if self._pelican_settings["PHOTO_WATERMARK_IMG"]:
            mark_image = PILImage.open(self._pelican_settings["PHOTO_WATERMARK_IMG"])
            mark_image_size = [
                watermark_layer.size[0] // image_reducer for size in mark_size
            ]
            mark_image_size = (
                self._pelican_settings["PHOTO_WATERMARK_IMG_SIZE"]
                if self._pelican_settings["PHOTO_WATERMARK_IMG_SIZE"]
                else mark_image_size
            )
            mark_image.thumbnail(mark_image_size, PILImage.LANCZOS)
            mark_position = [
                watermark_layer.size[i] - mark_image.size[i] - margin[i] for i in [0, 1]
            ]
            mark_position = (
                mark_position[0] - (text_size[0] // 2) + (mark_image_size[0] // 2),
                mark_position[1] - text_size[1],
            )

            if not self.is_alpha(mark_image):
                mark_image = mark_image.convert("RGBA")

            watermark_layer.paste(mark_image, mark_position, mark_image)

        watermark_layer = self.reduce_opacity(watermark_layer, opacity)
        image.paste(watermark_layer, (0, 0), watermark_layer)

        return image


class SrcSetImage(Image):
    """Image in a srcset."""

    def __init__(
        self,
        src,
        dst,
        spec: Optional[Dict[str, Any]] = None,
        is_thumb=False,
    ):
        self.descriptor = spec.get("srcset_descriptor", f"{spec['width']}w")

        dst_suffix = spec.get("srcset_extension")
        if dst_suffix is None:
            dst_suffix = self.descriptor

        dst = f"{dst}_{dst_suffix}"
        super().__init__(src=src, dst=dst, spec=spec, is_thumb=is_thumb)


class ImageSrcSet(list):
    """List of images in the srcset attribute of an HTML img-tag."""

    @property
    def html_srcset(self) -> str:
        """The string to put in srcset attribute of the img-tag."""
        items = []
        img: SrcSetImage
        for img in self:
            items.append(
                "{siteurl}/{filename} {descriptor}".format(
                    siteurl=pelican_settings["SITEURL"],
                    filename=img.web_filename,
                    descriptor=img.descriptor,
                )
            )
        return ", ".join(items)


class SourceImage:
    """A source image.

    - Detect mime-type
    - Load caption
    - Load exif information
    -
    """

    #: Dict to cache the source images, so we only have to process it once
    image_cache: ClassVar[Dict[str, "SourceImage"]] = {}

    def __init__(self, filename):
        #: filename of the image
        self.filename = filename
        #: mime-type of the image
        self.mimetype, _ = mimetypes.guess_type(filename)
        if not self.mimetype:
            raise InternalError(f"Unable to get MIME type of '{self.filename}'")
        file_type, _, image_type = self.mimetype.partition("/")
        if file_type != "image":
            raise FileExcluded(
                f"Skipe file '{self.filename}' because MIME type is '{file_type}' "
                "but must be not 'image'."
            )

        #: type of the image. Mostly the second part of the mime-type (jpeg, png, ...)
        self.type = image_type.lower()

        #: Internal caption object
        self._caption = Caption(source_image=self)

        #: Internal exif data
        self._exif = Exif(source_image=self)

        #: Internal exclude list
        self._excluded = ExcludeList(source_image=self)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(filename={self.filename},"
            f" mimetype={self.mimetype}, type={self.type})"
        )

    @property
    def caption(self) -> Optional[Caption]:
        """The caption of the image."""
        if self._caption.value is None:
            return None
        return self._caption

    @property
    def exif(self) -> Optional[Exif]:
        """Exif information."""
        if self._exif.value is None:
            return None
        return self._exif

    @property
    def is_excluded(self) -> bool:
        """Is the image excluded."""
        if self._excluded.value is None:
            return False
        return True

    def open(self) -> PILImage.Image:
        """Open the image with PIL/Pillow."""
        logger.debug(f"photos: Open file {self.filename}")
        return PILImage.open(self.filename)

    @classmethod
    def from_cache(cls, filename: str) -> "SourceImage":
        """Create a new SrcImage object or return it from the cache."""
        source_image = cls.image_cache.get(filename)
        if source_image is None:
            source_image = cls(filename=filename)
            cls.image_cache[filename] = source_image

        return source_image


def destroy_process_pool():
    global g_process_pool
    global g_process_pool_initialized

    if g_process_pool is not None:
        logger.info("Send close command to process pool ...")
        g_process_pool.close()
        logger.info("Waiting for processes to finish  ...")
        g_process_pool.join()
        g_process_pool = None

    g_process_pool_initialized = False


def get_process_pool():
    global g_process_pool
    global g_process_pool_initialized

    if g_process_pool_initialized:
        return g_process_pool

    resize_job_number: int = pelican_settings["PHOTO_RESIZE_JOBS"]

    if resize_job_number < 0:
        resize_job_number = 1

    if resize_job_number == 0:
        resize_job_number = os.cpu_count() + 1

    if g_process_pool is not None:
        logger.debug(
            f"Process pool in state {g_process_pool._state} and with"
            f" {len(g_process_pool._pool)} worker(s) already exists"
        )
        if g_process_pool._state != multiprocessing.pool.RUN:
            logger.warning("Existing pool is not in RUN state. Terminating pool.")
            g_process_pool.terminate()
            g_process_pool.join()
            g_process_pool = None

    if g_process_pool:
        g_process_pool_initialized = True
        return g_process_pool

    if resize_job_number == 1:
        logger.info("Process pool has been disabled, because we ony want 1 process")
    else:
        logger.info(f"Creating process pool with {resize_job_number} worker(s)")
        g_process_pool = multiprocessing.Pool(processes=resize_job_number)

    g_process_pool_initialized = True
    return g_process_pool


def initialized(pelican: Pelican):  # noqa: PLR0915
    """Initialize the default settings."""
    p = os.path.expanduser("~/Pictures")
    DEFAULT_CONFIG.setdefault("PHOTO_LIBRARY", p)
    DEFAULT_CONFIG.setdefault(
        "PHOTO_GALLERY",
        {
            "default": {
                "width": 1024,
                "height": 768,
                "type": "jpeg",
                "options": {"quality": 80},
            }
        },
    )
    DEFAULT_CONFIG.setdefault(
        "PHOTO_ARTICLE",
        {
            "default": {
                "width": 760,
                "height": 506,
                "type": "jpeg",
                "options": {"quality": 80},
            }
        },
    )
    DEFAULT_CONFIG.setdefault(
        "PHOTO_THUMB",
        {
            "default": {
                "width": 192,
                "height": 144,
                "type": "jpeg",
                "options": {"quality": 60},
            }
        },
    )
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

    if pelican:
        pelican.settings.setdefault("PHOTO_LIBRARY", p)
        pelican.settings.setdefault(
            "PHOTO_GALLERY",
            {
                "default": {
                    "width": 1024,
                    "height": 768,
                    "type": "jpeg",
                    "options": {"quality": 80},
                }
            },
        )
        pelican.settings.setdefault(
            "PHOTO_ARTICLE",
            {
                "default": {
                    "width": 760,
                    "height": 506,
                    "type": "jpeg",
                    "options": {"quality": 80},
                }
            },
        )
        pelican.settings.setdefault(
            "PHOTO_THUMB",
            {
                "default": {
                    "width": 192,
                    "height": 144,
                    "type": "jpeg",
                    "options": {"quality": 60},
                }
            },
        )
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
        pelican.settings.setdefault("PHOTO_GLOBAL_IMAGES", {})
        pelican.settings.setdefault("PHOTO_PROFILES", {})
        pelican.settings.setdefault(
            "PHOTO_EXIF_COPYRIGHT_AUTHOR", pelican.settings["AUTHOR"]
        )
        pelican.settings.setdefault(
            "PHOTO_FILE_EXTENSIONS", {"jpeg": "jpg", "webp": "webp"}
        )
        pelican.settings.setdefault("PHOTO_LIGHTBOX_GALLERY_ATTR", "data-lightbox")
        pelican.settings.setdefault("PHOTO_LIGHTBOX_CAPTION_ATTR", "data-title")

        pelican.settings.setdefault("PHOTO_INLINE_ENABLED", False)
        pelican.settings.setdefault(
            "PHOTO_INLINE_PATTERN",
            (
                r"(?is)"
                r"<(?P<tag>[^\s>]+)"
                r"(\s+[^>]+)?\s+"
                r"(?P<type>(gallery|image|lightbox))\s*=\s*"
                r"(?P<quote>[\"'])"
                r"(?P<name>.*?)"
                r"(?P=quote)"
                r"([^>]+)?"
                r"(/>|>.*?</(?P=tag)>)"
            ),
        )
        pelican.settings.setdefault("PHOTO_INLINE_PARSE_HTML", True)
        pelican.settings.setdefault("PHOTO_INLINE_GALLERY_ENABLED", False)
        pelican.settings.setdefault(
            "PHOTO_INLINE_GALLERY_PATTERN", r"gallery::(?P<gallery_name>[/{}\w_-]+)"
        )
        pelican.settings.setdefault("PHOTO_INLINE_GALLERY_TEMPLATE", "inline_gallery")
        pelican.settings.setdefault("PHOTO_INLINE_IMAGE_TEMPLATE", "inline_image")
        pelican.settings.setdefault("PHOTO_INLINE_LIGHTBOX_TEMPLATE", "inline_lightbox")
        pelican.settings.setdefault(
            "PHOTO_DEFAULT_IMAGE_OPTIONS", {"jpeg": {"optimize": True}}
        )

        for name in ("PHOTO_ARTICLE", "PHOTO_GALLERY", "PHOTO_THUMB"):
            if isinstance(pelican.settings[name], (list, tuple)):
                logger.info(f"Converting legacy config to new values: {name}")
                pelican.settings[name] = {
                    "default": {
                        "width": pelican.settings[name][0],
                        "height": pelican.settings[name][1],
                        "type": "jpeg",
                        "options": {"quality": pelican.settings[name][2]},
                    }
                }

        pelican.settings.setdefault("PHOTO_RESULT_IMAGE_AVERAGE_COLOR", False)
        pelican.settings.setdefault("PHOTO_GLOBAL_IMAGES_PROCESSED", {})
        pelican.settings.setdefault("PHOTO_PROFILING_ENABLED", False)

    global pelican_settings
    pelican_settings = pelican.settings
    global pelican_output_path
    pelican_output_path = pelican.output_path
    global g_profiles
    g_profiles = {}
    default_profile = Profile(
        name="default",
        config={
            "images": {
                "article": {"specs": pelican_settings["PHOTO_ARTICLE"]},
                "gallery": {"specs": pelican_settings["PHOTO_GALLERY"]},
                "thumb": {"specs": pelican_settings["PHOTO_THUMB"]},
            }
        },
    )
    g_profiles["default"] = default_profile
    for profile_name, profile_config in pelican_settings["PHOTO_PROFILES"].items():
        if isinstance(profile_config, dict):
            g_profiles[profile_name] = Profile(
                name=profile_name,
                config=profile_config,
                default_profile=default_profile,
            )
    for profile_name, profile_config in pelican_settings["PHOTO_PROFILES"].items():
        if isinstance(profile_config, str):
            g_profiles[profile_name] = g_profiles[profile_config]


@measure_time
def enqueue_image(img: Image) -> Image:
    """Add the image to the resize list.

    If an image with the same destination filename and the same specifications does
    already exist it will return this instead.

    """
    if img.dst not in g_image_cache:
        g_image_cache[img.dst] = img
        g_image_queue.append(img)
    elif (
        g_image_cache[img.dst].source_image != img.source_image
        or g_image_cache[img.dst].spec != img.spec
    ):
        raise InternalError(
            "Resize conflict for {}, {}-{} is not {}-{}".format(  # noqa: UP032
                img.dst,
                g_image_cache[img.dst].source_image.filename,
                g_image_cache[img.dst].spec,
                img.source_image.filename,
                img.spec,
            )
        )
    return img


def build_license(license, author):
    year = datetime.datetime.now().year
    license_file = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "licenses.json"
    )

    with open(license_file) as data_file:
        licenses = json.load(data_file)

    if any(license in k for k in licenses):
        return licenses[license]["Text"].format(
            Author=author, Year=year, URL=licenses[license]["URL"]
        )

    return f"Copyright {year} {author}, All Rights Reserved"


def process_image_process_wrapper(image: Image):
    """Create a wrapper to call an object function in the pool process."""
    try:
        return image.process()
    except Exception as e:  # noqa: BLE001 -- we are just warning here
        logger.error(f"photos: {e}")
        logger.warning("photos: An exception occurred", exc_info=e)


@measure_time
def process_image_queue():
    """Launch the jobs to process the images in the resize queue."""
    global g_image_queue

    logger.info(f"Found {len(g_image_queue)} images in resize queue")
    if len(g_image_queue) == 0:
        return

    results = {}
    image_queue = g_image_queue
    g_image_queue = []

    process_pool = get_process_pool()
    if process_pool is None:
        for image in image_queue:
            result = image.process()
            if result:
                results.update(dict((result,)))
    else:
        results = dict(
            filter(
                lambda v: v is not None,
                process_pool.imap_unordered(process_image_process_wrapper, image_queue),
            )
        )

    logger.info(f"Applying results for {len(results)} generated images")
    for k, result_info in results.items():
        g_image_cache[k].apply_result_info(result_info)


@measure_time
def detect_inline_images(content: pelican.contents.Content):  # noqa: PLR0912
    """Find images in the generated content and replace them with the processed images."""  # noqa: E501
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
    """.format(pelican_settings["INTRASITE_LINK_REGEX"])
    hrefs = re.compile(regex, re.X)

    inline_images = {}
    if not content._content or not (
        "{photo}" in content._content or "{lightbox}" in content._content
    ):
        return inline_images
    # content._content = hrefs.sub(replacer, content._content)
    for m in hrefs.finditer(content._content):
        tag = m.group("tag")
        what = m.group("what")
        value = m.group("value")

        if what not in ("photo", "lightbox"):
            # ToDo: Log unsupported type
            continue

        if value.startswith("/"):
            value = value[1:]

        if tag == "img":
            parser = HTMLImgParser()
            parser.feed(m.group())

            img_classes = []
            for img_attr_name, img_attr_value in parser.img_attrs:
                if img_attr_name == "class":
                    img_classes = [v.strip() for v in img_attr_value.split(" ")]
            profile = find_profile(img_classes)

            if what == "photo":
                try:
                    img = ContentImage(
                        filename=value,
                        profile=profile,
                    )
                except FileNotFound as e:
                    logger.error(f"photos: {e!s}")
                    continue

            elif what == "lightbox":
                try:
                    img = ContentImageLightbox(
                        filename=value,
                        profile=profile,
                    )
                except FileNotFound as e:
                    logger.error(f"photos: {e!s}")
                    continue
            else:
                logger.warning(f"Unable to detect type '{what}' for '{m.group()}'")
                continue

            inline_images[m.group()] = {
                "parsed_attrs": parser.img_attrs,
                "match": m,
                "image": img,
            }

        elif what == "photo":
            try:
                img = ContentImage(filename=value)
            except FileNotFound as e:
                logger.error(f"photos: {e!s}")
                continue

            inline_images[m.group()] = {
                "match": m,
                "image": img,
            }

        # else:
        #  logger.error("photos: No photo %s", value)

    return inline_images


def galleries_string_decompose(gallery_string) -> List[Dict[str, Any]]:
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

    logger.error(f"Unexpected gallery location format! \n{pprint.pformat(galleries)}")
    return []


@measure_time
def process_content_galleries(
    content: Union[Article, Page],
    location,
    profile_name: Optional[str] = None,
) -> List[Gallery]:
    """Process all galleries attached to an article or page.

    :param content: The content object
    :param location: Galleries
    :param profile_name:
    """
    photo_galleries = []

    galleries = galleries_string_decompose(location)

    for gallery_name in galleries:
        try:
            gallery = Gallery(content, gallery_name, profile_name=profile_name)
            photo_galleries.append(gallery)
        except GalleryNotFound as e:
            logger.error(f"photos: {e!s}")

    return photo_galleries


@measure_time
def detect_inline_galleries(content: Union[Article, Page]):
    """Find galleries specified as inline gallery."""
    inline_galleries = {}
    if pelican_settings["PHOTO_INLINE_GALLERY_ENABLED"]:
        gallery_strings = re.finditer(
            pelican_settings["PHOTO_INLINE_GALLERY_PATTERN"], content._content
        )
        for m in gallery_strings:
            inline_galleries[str(m.group())] = process_content_galleries(
                content, m.group("gallery_name")
            )

    return inline_galleries


@measure_time
def detect_inline_contents(content: Union[Article, Page]):
    """Find inline galleries, images, ..."""
    if not pelican_settings["PHOTO_INLINE_ENABLED"]:
        return {}

    inline_contents = {}
    content_strings = re.finditer(
        pelican_settings["PHOTO_INLINE_PATTERN"], content._content
    )
    for m in content_strings:
        profile_name = None
        html_attributes = {}
        if pelican_settings["PHOTO_INLINE_PARSE_HTML"]:
            parser = HTMLTagParser()
            parser.feed(m.group())
            html_attributes = parser.tag_attrs
            profile_name = html_attributes.get("profile")

        if m.group("type") == "gallery":
            inline_contents[str(m.group())] = InlineContentData(
                m.group("type"),
                process_content_galleries(
                    content=content,
                    location=m.group("name"),
                    profile_name=profile_name,
                ),
                html_attributes,
            )

        elif m.group("type") == "image":
            inline_contents[str(m.group())] = InlineContentData(
                m.group("type"),
                ContentImage(
                    filename=m.group("name"),
                    profile_name=profile_name,
                ),
                html_attributes,
            )
        elif m.group("type") == "lightbox":
            inline_contents[str(m.group())] = InlineContentData(
                m.group("type"),
                ContentImageLightbox(
                    filename=m.group("name"),
                    profile_name=profile_name,
                ),
                html_attributes,
            )
        else:
            logger.error(f"Unsupported type '{m.group('type')}' in '{m.group()}")
    return inline_contents


def detect_meta_galleries(content: Union[Article, Page]):
    """Find galleries specified in the meta data or as inline gallery."""
    if "gallery" in content.metadata:
        gallery = content.metadata.get("gallery")
        if gallery.startswith("{photo}") or gallery.startswith("{filename}"):
            content.photo_gallery = process_content_galleries(content, gallery)
        elif gallery:
            logger.error(f"photos: Gallery tag not recognized: {gallery}")


def detect_meta_images(content: pelican.contents.Content):
    """Look for article or page photos specified in the meta data.

    Find images in the generated content and replace them with the processed images.
    """
    image = content.metadata.get("image", None)
    if image:
        if image.startswith("{photo}") or image.startswith("{filename}"):
            try:
                content.photo_image = ArticleImage(content=content, filename=image)
            except (FileNotFound, InternalError) as e:
                logger.error(f"photo: {e!s}")
        else:
            logger.error(f"photos: Image tag not recognized: {image}")

    images = {}
    for meta_name, meta_value in content.metadata.items():
        meta_prefix, _, meta_image_name = meta_name.partition("_")
        if meta_image_name and meta_prefix.lower().strip() == "image":
            images[meta_image_name.strip()] = get_image_from_string(
                url_string=meta_value, default_image_class=ContentImage
            )

    if len(images) > 0:
        content.photo_images = images


@measure_time
def replace_inline_contents(content, inline_contents):
    for content_string, content_info in inline_contents.items():
        image = content_info.image
        template_values = {
            "content": content,
            "html_attributes": content_info.html_attributes,
        }

        profile = None

        if content_info.type == "gallery":
            template_values["default_template_name"] = pelican_settings[
                "PHOTO_INLINE_GALLERY_TEMPLATE"
            ]
            template_values["galleries"] = image
            # We use the profile from the first gallery
            if len(image) > 0:
                profile = image[0].profile
            else:
                logger.warning(
                    f"No gallery found for '{content_string}' in '{content.filename}'"
                )
        elif content_info.type == "image":
            template_values["default_template_name"] = pelican_settings[
                "PHOTO_INLINE_IMAGE_TEMPLATE"
            ]
            template_values["image"] = image
            profile = image.profile
        elif content_info.type == "lightbox":
            template_values["default_template_name"] = pelican_settings[
                "PHOTO_INLINE_LIGHTBOX_TEMPLATE"
            ]
            template_values["lightbox_image"] = image
            profile = image.profile
        else:
            logger.error(f"Unable to handle type '{content_info.type}")
            continue

        if isinstance(content, Article):
            template_values["article"] = content
        elif isinstance(content, Page):
            template_values["page"] = content

        if profile:
            content._content = content._content.replace(
                content_string, profile.render_template(**template_values)
            )
        else:
            logger.warning(
                f"Unable to find profile for '{content_string}'"
                f" in '{content.filename}'. Removing from output."
                " Have a look for other errors or warnings to fix this issue."
            )
            content._content = content._content.replace(content_string, "")


@measure_time
def replace_inline_galleries(content, inline_galleries):
    for gallery_string, galleries in inline_galleries.items():
        template = g_generator.get_template(
            pelican_settings["PHOTO_INLINE_GALLERY_TEMPLATE"]
        )
        template_values = {
            "galleries": galleries,
            "SITEURL": pelican_settings["SITEURL"],
        }
        if isinstance(content, Article):
            template_values["article"] = content
        elif isinstance(content, Page):
            template_values["page"] = content

        content._content = content._content.replace(
            gallery_string, template.render(**template_values)
        )


@measure_time
def replace_inline_images(content, inline_images):
    for image_string, image_info in inline_images.items():
        m = image_info["match"]
        image = image_info["image"]
        parsed_attrs = image_info.get("parsed_attrs")
        profile = image.profile

        what = m.group("what")
        value = m.group("value")
        tag = m.group("tag")

        if value.startswith("/"):
            value = value[1:]

        extra_attributes = ""
        html_img_attributes = profile.thumb_html_img_attributes
        if html_img_attributes:
            for prof_attr_name, prof_attr_value in html_img_attributes.items():
                extra_attributes += ' {}="{}"'.format(  # noqa: UP032
                    prof_attr_name, prof_attr_value.format(i=image)
                )

        if profile.has_template:
            content._content = content._content.replace(
                image_string,
                profile.render_template(
                    content=content,
                    image=image,
                    match=m,
                    parsed_attrs=dict(parsed_attrs),
                ),
            )

        elif what == "photo":
            content._content = content._content.replace(
                image_string,
                "".join(
                    (
                        "<",
                        m.group("tag"),
                        m.group("attrs_before"),
                        m.group("src"),
                        "=",
                        m.group("quote"),
                        "{siteurl}/{filename}".format(
                            siteurl=pelican_settings["SITEURL"],
                            filename=image.image.web_filename,
                        ),
                        m.group("quote"),
                        extra_attributes,
                        m.group("attrs_after"),
                    )
                ),
            )

        elif what == "lightbox" and tag == "img":
            lightbox_attr_list = [""]

            gallery_name = value.split("/")[0]
            lightbox_attr_list.append(
                '{}="{}"'.format(
                    pelican_settings["PHOTO_LIGHTBOX_GALLERY_ATTR"], gallery_name
                )
            )

            if image.caption:
                lightbox_attr_list.append(
                    '{}="{}"'.format(
                        pelican_settings["PHOTO_LIGHTBOX_CAPTION_ATTR"],
                        str(image.caption),
                    )
                )

            lightbox_attrs = " ".join(lightbox_attr_list)

            content._content = content._content.replace(
                image_string,
                "".join(
                    (
                        "<a href=",
                        m.group("quote"),
                        "{siteurl}/{filename}".format(
                            siteurl=pelican_settings["SITEURL"],
                            filename=image.image.web_filename,
                        ),
                        m.group("quote"),
                        lightbox_attrs,
                        "><img",
                        m.group("attrs_before"),
                        "src=",
                        m.group("quote"),
                        "{siteurl}/{filename}".format(
                            siteurl=pelican_settings["SITEURL"],
                            filename=image.thumb.web_filename,
                        ),
                        m.group("quote"),
                        extra_attributes,
                        m.group("attrs_after"),
                        "</a>",
                    )
                ),
            )


def handle_signal_finalized(pelican: Pelican):
    destroy_process_pool()


def handle_signal_generator_init(generator):
    global g_generator
    g_generator = generator


@measure_time
def handle_signal_content_object_init(content: pelican.contents.Content):
    if not isinstance(content, (Article, Page)):
        return
    inline_images = detect_inline_images(content)
    inline_galleries = detect_inline_galleries(content)
    inline_contents = detect_inline_contents(content)
    process_image_queue()
    replace_inline_images(content, inline_images)
    replace_inline_galleries(content, inline_galleries)
    replace_inline_contents(content, inline_contents)


@measure_time
def handle_signal_all_generators_finalized(
    generators: List[pelican.generators.Generator],
):
    global_images = {}
    for name, config in pelican_settings["PHOTO_GLOBAL_IMAGES"].items():
        try:
            global_images[name] = GlobalImage(
                filename=config["filename"],
                profile_name=config.get("profile"),
            )
        except (FileNotFound, InternalError) as e:
            logger.error(f"photo: {e!s}")

    pelican_settings["PHOTO_GLOBAL_IMAGES_PROCESSED"].update(global_images)

    for generator in generators:
        if isinstance(generator, ArticlesGenerator):
            article: Article
            article_lists: List[List[Article]] = [
                generator.articles,
                generator.translations,
                generator.drafts,
                generator.drafts_translations,
            ]
            # Support for hidden articles has been added in Pelican 4.7
            if hasattr(generator, "hidden_articles"):
                article_lists.extend(
                    [generator.hidden_articles, generator.hidden_translations]
                )
            for article in itertools.chain.from_iterable(article_lists):
                detect_meta_images(article)
                detect_meta_galleries(article)
                article.photo_global_images = global_images
        elif isinstance(generator, PagesGenerator):
            page: Page
            for page in itertools.chain(
                generator.pages,
                generator.translations,
                generator.hidden_pages,
                generator.hidden_translations,
                generator.draft_pages,
                generator.draft_translations,
            ):
                detect_meta_images(page)
                detect_meta_galleries(page)
                page.photo_global_images = global_images

    process_image_queue()


def register():
    """Use the new style of registration based on GitHub Pelican issue #314."""
    signals.initialized.connect(initialized)
    signals.generator_init.connect(handle_signal_generator_init)
    signals.content_object_init.connect(handle_signal_content_object_init)
    signals.all_generators_finalized.connect(handle_signal_all_generators_finalized)
    signals.finalized.connect(handle_signal_finalized)
