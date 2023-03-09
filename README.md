Photos: A Plugin for Pelican
============================

[![Build Status](https://img.shields.io/github/workflow/status/pelican-plugins/photos/build)](https://github.com/pelican-plugins/photos/actions)
[![PyPI Version](https://img.shields.io/pypi/v/pelican-photos)](https://pypi.org/project/pelican-photos/)
![License](https://img.shields.io/pypi/l/pelican-photos?color=blue)

Use Photos to add a photo or a gallery of photos to an article, or to include photos in the body text. Photos are kept separately, as an organized library of high resolution photos, and resized as needed.

Installation
------------

This plugin can be installed via:

    python -m pip install pelican-photos

The plug-in requires `Pillow`: the Python Imaging Library and optionally `Piexif`, whose installation are outside the scope of this document.

Usage
-----

The plug-in resizes the referred photos, and generates thumbnails for galleries and associated photos, based on the following configuration and default values:

`PHOTO_LIBRARY = "~/Pictures"`
:	Absolute path to the folder where the original photos are kept, organized in sub-folders.

`PHOTO_GALLERY = (1024, 768, 80)`
:	For photos in galleries, maximum width and height, plus JPEG quality as a percentage. This would typically be the size of the photo displayed when the reader clicks a thumbnail.

`PHOTO_ARTICLE = (760, 506, 80)`
:	For photos associated with articles, maximum width, height, and quality. The maximum size would typically depend on the needs of the theme. 760px is suitable for the theme `notmyidea`.

`PHOTO_THUMB = (192, 144, 60)`
:	For thumbnails, maximum width, height, and quality.

`PHOTO_SQUARE_THUMB = False`
:	Crops thumbnails to make them square.

`PHOTO_RESIZE_JOBS = 5`
: Number of parallel resize jobs to be run. Defaults to 1.

- `-1` = Do not use the Python multiprocessing module and enable additional debugging
- `0` = Try to detect cpu count and start `CPU_COUNT + 1` processes
- `1` = Use 1 process
- `2` = Use 2 processes
- `3` = Use 3 processes
- ...

`PHOTO_WATERMARK = True`
: Adds a watermark to all photos in articles and pages. Defaults to using your site name.

`PHOTO_WATERMARK_TEXT' = SITENAME`
: Allow the user to change the watermark text or remove it completely. By default it uses [SourceCodePro-Bold](http://www.adobe.com/products/type/font-information/source-code-pro-readme.html) as the font.

`PHOTO_WATERMARK_IMG = ''`
: Allows the user to add an image in addition to or as the only watermark. Set the variable to the location.

**The following features require the piexif library**
`PHOTO_EXIF_KEEP = True`
: Keeps the exif of the input photo.

`PHOTO_EXIF_REMOVE_GPS = True`
: Removes any GPS information from the files exif data.

`PHOTO_EXIF_COPYRIGHT = 'COPYRIGHT'`
: Attaches an author and a license to the file. Choices include:
	- `COPYRIGHT`: Copyright
	- `CC0`: Public Domain
	- `CC-BY-NC-ND`: Creative Commons Attribution-NonCommercial-NoDerivatives
	- `CC-BY-NC-SA`: Creative Commons Attribution-NonCommercial-ShareAlike
	- `CC-BY`: Creative Commons Attribution
	- `CC-BY-SA`: Creative Commons Attribution-ShareAlike
	- `CC-BY-NC`: Creative Commons Attribution-NonCommercial
	- `CC-BY-ND`: Creative Commons Attribution-NoDerivatives

`PHOTO_EXIF_COPYRIGHT_AUTHOR = 'Your Name Here'`
: Adds an author name to the photo's exif and copyright statement. Defaults to `AUTHOR` value from the `pelicanconf.py`

`PHOTO_INLINE_GALLERY_ENABLED`
: Enable inline gallery processing. (Default: False)
With this feature, a gallery could be placed everywhere in an article.

`PHOTO_INLINE_GALLERY_PATTERN`
: The pattern to look for. The ```gallery_name``` is used to find the right gallery
and the optional ``options`` is used to transmit options to the ``inline_gallery.html``
template file.

Defaults:

```python
r"gallery(\[(?P<options>[\w,=]+)\])?::(?P<gallery_name>[/{}\w_-]+)"
```

`PHOTO_INLINE_GALLERY_TEMPLATE`
: Template to render the inline gallery.

Default: ```"inline_gallery"```

`PHOTO_DEFAULT_IMAGE_OPTIONS`
: Default keyword options to pass to the PIL.Image.save() function.
You can find all available options in the [Pillow: Image file formats](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#image-file-formats) section.

Defaults:
```python
{
	"jpeg": {
		"optimize": True
	}
}
```

`PHOTO_RESULT_IMAGE_AVERAGE_COLOR = False`
: Calculate the average color for an result image. This can be used to provide a default background color while using lazy loading.

`PHOTO_PROFILING_ENABLED = False`
: Enable profiling of function calls and measure the time. You have to run pelican in debug mode to see the results.

The plug-in automatically resizes the photos and publishes them to the following output folder:

    ./output/photos

**WARNING:** The plug-in can take hours to resize 40,000 photos, therefore, photos and thumbnails are only generated once. Clean the output folders to regenerate the resized photos again.

## How to use

Maintain an organized library of high resolution photos somewhere on disk, using folders to group related images. The default path `~/Pictures` is convenient for Mac OS X users.

* To create a gallery of photos, add the metadata field `gallery: {photo}folder` to an article. To simplify the transition from the plug-in Gallery, the syntax `gallery: {filename}folder` is also accepted.
* You can now have multiple galleries. The galleries need to be seperated by a comma in the metadata field. The syntax is gallery: `{photo}folder, {photo}folder2`. You can also add titles to your galleries. The syntax is: `{photo}folder, {photo}folder2{This is a title}`. Using the following example the first gallery would have the title of the folder location and the second would have the title `This is a tile.`
* To use an image in the body of the text, just use the syntax `{photo}folder/image.jpg` instead of the usual `{filename}/images/image.jpg`.
* To use an image in the body of the text, which can be used with [Lightbox](http://lokeshdhakar.com/projects/lightbox2/) just use the syntax `{lightbox}folder/image.jpg`. For use with other implementations, the gallery and caption attribute names can be set with `PHOTO_LIGHTBOX_GALLERY_ATTR` and `PHOTO_LIGHTBOX_CAPTION_ATTR`.
* To associate an image with an article, add the metadata field `image: {photo}folder/image.jpg` to an article. Use associated images to improve navigation. For compatibility, the syntax `image: {filename}/images/image.jpg` is also accepted.

### Exif, Captions, and Blacklists
Folders of photos may optionally have three text files, where each line describes one photo. You can use the `#` to comment out a line. Generating these optional files is left as an exercise for the reader (but consider using Phil Harvey's [exiftool](http://www.sno.phy.queensu.ca/~phil/exiftool/)). See below for one method of extracting exif data.

`exif.txt`
:	Associates compact technical information with photos, typically the camera settings. For example:

	best.jpg: Canon EOS 5D Mark II - 20mm f/8 1/250s ISO 100
	night.jpg: Canon EOS 5D Mark II - 47mm f/8 5s ISO 100
	# new.jpg: Canon EOS 5D Mark II - 47mm f/8 5s ISO 100

`captions.txt`
:	Associates comments with photos. For example:

	best.jpg: My best photo ever! How lucky of me!
	night.jpg: Twilight over the dam.
	# new.jpg: My new photo blog entry is not quite ready.

`blacklist.txt`
: Skips photos the user does not want to include. For example:

	this-file-will-be-skipped.jpg
	this-one-will-be-skipped-too.jpg
	# but-this-file-will-NOT-be-skipped.jpg
	this-one-will-be-also-skipped.jpg


Here is an example Markdown article that shows the four use cases:

	title: My Article
	gallery: {photo}favorite
	image: {photo}favorite/best.jpg

	Here are my best photos, taken with my favorite camera:
	![]({photo}mybag/camera.jpg).
	![]({lightbox}mybag/flash.jpg).

The default behavior of the Photos plugin removes the exif information from the file. If you would like to keep the exif information, you can install the `piexif` library for python and add the following settings to keep some or all of the exif information. This feature is not a replacement for the `exif.txt` feature but in addition to that feature. This feature currently only works with jpeg input files.

## How to change the Jinja templates

The plugin provides the following variables to your templates:

`article.photo_image`
:	For articles with an associated photo, a tuple with the following information:

* The filename of the original photo.
* The output path to the generated photo.
* The output path to the generated thumbnail.

For example, modify the template `article.html` as shown below to display the associated image before the article content:

```html
<div class="entry-content">
	{% if article.photo_image %}<img src="{{ SITEURL }}/{{ article.photo_image[1] }}" />{% endif %}
	{% include 'article_infos.html' %}
	{{ article.content }}
</div><!-- /.entry-content -->
```

`article.photo_gallery`
:	For articles with a gallery, a list of the photos in the gallery. Each item in the list is a tuple with five elements:

* The title of the gallery
* The filename of the original photo.
* The output path to the generated photo.
* The output path to the generated thumbnail.
* The EXIF information of the photo, as read from the file `exif.txt`.
* The caption of the photo, as read from `captions.txt`.

For example, add the following to the template `article.html` to add the gallery as the end of the article:

```html
{% if article.photo_gallery %}
<div class="gallery">
		{% for title, gallery in article.photo_gallery %}
			<h1>{{ title }}</h1>
				{% for name, photo, thumb, exif, caption in gallery %}
						<a href="{{ SITEURL }}/{{ photo }}" title="{{ name }}" exif="{{ exif }}" caption="{{ caption }}"><img src="{{ SITEURL }}/{{ thumb }}"></a>
				{% endfor %}
		{% endfor %}
</div>
{% endif %}
```

For example, add the following to the template `index.html`, inside the `entry-content`, to display the thumbnail with a link to the article:

```html
{% if article.photo_image %}<a href="{{ SITEURL }}/{{ article.url }}"><img src="{{ SITEURL }}/{{ article.photo_image[2] }}"
	style="display: inline; float: right; margin: 2px 0 2ex 4ex;" /></a>
{% endif %}
```

## The new optional templates files for inline galleries, images and lightboxes

There are three corresponding template files.

### Inline gallery

When the variable `PHOTO_INLINE_GALLERY_TEMPLATE` is set to true,
a gallery could be placed everywhere in an article.
An inline gallery is introduced in Markdown or in reStructuredText as:

```
  gallery::{photo}mygallery
```

The template file to render inline galleries is loaded.
The name of this file is by default ``"inline_gallery.html"```
and could be changed with the  `PHOTO_INLINE_GALLERY_TEMPLATE`  variable.
The substituted variable is `gallery` which is the previous tuple with five elements.
An example of this file is:
```html
<div class="gallery">
    {% for title, gallery in galleries%}
        <h1>{{ title }}</h1>
            {% for name, photo, thumb, exif, caption in gallery %}
                    <a href="{{ SITEURL }}/{{ photo }}" title="{{ name }}" exif="{{ exif }}" caption="{{ caption }}"><img src="{{ SITEURL }}/{{ thumb }}"></a>
            {% endfor %}
    {% endfor %}
</div>
```

The `PHOTO_INLINE_GALLERY_PATTERN`
variable in the ``pelicanconf.py`` file defined the pattern to look for both the
```gallery_name``` and the ``options``, used to transmit options
to the ``inline_gallery.html`.
Its default value is

```python
r"gallery(\[(?P<options>[\w,=]+)\])?::(?P<gallery_name>[/{}\w_-]+)"
```
An inline gallery transmitting an option
is introduced in Markdown or in reStructuredText as:

```
  gallery[reverse]::{photo}mygallery
```

The ``"inline_gallery.html"``` template file is then modified
accordingly to take into accound this option:

```html
<div class="gallery">
    {% if options and options['reverse'] %}
      {% set use_reverse = True %}
    {% else %}
      {% set use_reverse = False %}
    {% endif %}
    {% for title, gallery in galleries%}
        <h1>{{ title }}</h1>
            {% for name, photo, thumb, exif, caption in (gallery | sort(reverse=use_reverse,attribute="filename")) %}
                    <a href="{{ SITEURL }}/{{ photo }}" title="{{ name }}" exif="{{ exif }}" caption="{{ caption }}"><img src="{{ SITEURL }}/{{ thumb }}"></a>
            {% endfor %}
    {% endfor %}
</div>
```

The ``options`` variable is a python ``dict``:
it it is defined as a coma-separated list of ``keys``
and ``value`` pairs, e.g.

```
  gallery[reverse,width=20em]::{photo}mygallery
```

When there is no explicit value, the value is a ``bool``, set to ``True``.
When the value is explicitely ``False``, it is also converted to ``bool``,
otherwise it is left as a ``string``.

### Inline image

An inlined image is introduced in Markdown as:

```
  ![alttext]({image}mygallery/myimage.jpg){height=1.2em}
```

or in reStructuredText as:

```
  .. |mylabel| image:: {photo}mygallery/myimage.jpg
            :height: 1.2em

  And then, it could be inserted |mylabel| everywhere in the text.
```

Inlined images are rendered by default using an internal template scheme.
This could be customized by providing a template file.
The name of this file is by default ``"inline_image.html"```
and could be changed with the  `PHOTO_INLINE_IMAGE_TEMPLATE`  variable.
When this file is founded, it replaces the default internal template.
An example of this file is:
```html
  <{{ tag }} {{ attributes_before }} {{ src }}="{{ SITEURL }}/{{ image_article }}" {{ attributes_after }}
```
The content of this example file corresponds exactly to the internal default behavior
and could be customized. The substituted variables are:

  * `SITEURL` : the propagated variable from `pelicanconf.py` or ``"."``
     when `RELATIVE_URLS` is set to `True`.
  * `gallery_name` : the title of the gallery
  * `image_source` : the output path to the original photo
  * `image_article` : the output path to the generated photo
  * `exif` : the EXIF information of the photo, as read from the file `exif.txt`
  * `caption` : the caption of the photo, as read from `captions.txt`
  * `tag` : the tag, e.g. `img`
  * `src` : the source keyword, e.g. `src`
  * `attributes_before` : attribute list that comes before the `src`
  * `attributes_after` : attribute list that comes after the `src`
  * `extra_attributes` : others attributes

### Inline lightbox

Similarly to inlined images, an inlined lightbox is introduced in Markdown as:

```
  ![alttext]({lightbox}mygallery/myimage.jpg){width=15%}
```

or in reStructuredText as:

```
  .. |mylabel| image:: {lightbox}mygallery/myimage.jpg
            :width: 15%

  And then, it could be inserted as:

  |mylabel|
```

Inlined lighboxes are also rendered by default using an internal template scheme
and this could be customized by providing a template file.
The name of this file is by default ``"inline_lighbox.html"```
and could be changed with the  `PHOTO_INLINE_LIGHTBOX_TEMPLATE`  variable.
When this file is founded, it replaces the default internal template.
An example of this file is:
```html
<a href="{{ SITEURL }}/{{ image_source }}" data-lightbox="{{ gallery_name}}" data-title="{{ caption}}">
  <{{ tag }} {{ attributes_before }} {{ src }}="{{ SITEURL }}/{{ image_thumb }}" {{ attributes_after }}
</a>
```

The content of this example file corresponds exactly to the internal default behavior
and could be customized. The substituted variables are the same as for the
inlined images, with, in addition:

  * `image_gthumb` : the output path to the generated thumbnail.

## How to make the gallery lightbox

There are several JavaScript libraries that display a list of images as a lightbox. The example below uses [Magnific Popup](http://dimsemenov.com/plugins/magnific-popup/), which allows the more complex initialization needed to display both the filename, the compact technical information, and the caption. The solution would be simpler if photos did not show any extra information.

Copy the files `magnific-popup.css` and `magnific-popup.js` to the root of your Pelican template.

Add the following to the template `base.html`, inside the HTML `head` tags:

```html
{% if (article and article.photo_gallery) or (articles_page and articles_page.object_list[0].photo_gallery) %}
	<link rel="stylesheet" href="{{ SITEURL }}/{{ THEME_STATIC_DIR }}/magnific-popup.css">
{% endif %}
```

Add the following to the template `base.html`, before the closing HTML `</body>` tag:

```JavaScript
{% if (article and article.photo_gallery) or (articles_page and articles_page.object_list[0].photo_gallery) %}
<!-- jQuery 1.7.2+ or Zepto.js 1.0+ -->
<script src="http://ajax.googleapis.com/ajax/libs/jquery/1.9.1/jquery.min.js"></script>

<!-- Magnific Popup core JS file -->
<script src="{{ SITEURL }}/{{ THEME_STATIC_DIR }}/magnific-popup.js"></script>
<script>
$('.gallery').magnificPopup({
	delegate: 'a',
	type: 'image',
	gallery: {
		enabled: true,
		navigateByImgClick: true,
		preload: [1,2]
	},
	image: {
		titleSrc: function(item) {
			if (item.el.attr('caption') && item.el.attr('exif')) {
				return (item.el.attr('caption').replace(/\\n/g, '<br />') +
					'<small>' + item.el.attr('title') + ' - ' + item.el.attr('exif') + '</small>');
			}
		return item.el.attr('title') + '<small>' + item.el.attr('exif') + '</small>';
	} }
});
</script>
{% endif %}
```

## How to make a Bootstrap Carousel

If you are using bootstrap, the following code is an example of how one could create a carousel.

```html
{% if article.photo_gallery %}
  {% for title, gallery in article.photo_gallery %}
    <h1>{{ title }}</h1>
    <div id="carousel-{{ loop.index }}" class="carousel slide">
      <ol class="carousel-indicators">
          {% for i in range(0, gallery|length) %}
          <li data-target="#carousel-{{ loop.index }}" data-slide-to="{{ i }}" {% if i==0 %} class="active" {% endif %}></li>
          {% endfor %}
      </ol>
      <div class="carousel-inner">
        {% for name, photo, thumb, exif, caption in gallery %}
          {% if loop.first %}
            <div class="item active">
          {% else %}
            <div class="item">
          {% endif %}
          <img src="{{ SITEURL }}/{{ photo }}" exif="{{ exif }}" alt="{{ caption }}">
          <div class="carousel-caption">
              <h5>{{ caption }}</h5>
          </div> <!-- carousel-caption -->
        </div> <!-- item -->
        {% endfor %}
      </div> <!-- carousel-inner -->
      <a class="left carousel-control" href="#carousel-{{ loop.index }}" data-slide="prev">
        <span class="glyphicon glyphicon-chevron-left"></span>
      </a>
      <a class="right carousel-control" href="#carousel-{{ loop.index }}" data-slide="next">
        <span class="glyphicon glyphicon-chevron-right"></span>
      </a>
    </div> <!-- closes carousel-{{ loop.index }} -->
    {% endfor %}
{% endif %}
```

## Exiftool example

You can add the following stanza to your fab file if you are using `fabric` to generate the appropriate text files for your galleries. You need to set the location of `Exiftool` control files.

```Python
def photo_gallery_gen(location):
    """Create gallery metadata files."""
    local_path = os.getcwd() + 'LOCATION OF YOUR EXIF CONTROL FILES'
    with lcd(location):
        local("exiftool -p {fmt_path}/exif.fmt . > exif.txt".format(
            fmt_path=local_path))
        local("exiftool -p {fmt_path}/captions.fmt . > captions.txt".format(
            fmt_path=local_path))

```

`captions.fmt` example file

```
$FileName: $Description
```

`exif.fmt` example file

```
$FileName: $CreateDate - $Make $Model Stats:(f/$Aperture, ${ShutterSpeed}s, ISO $ISO Flash: $Flash) GPS:($GPSPosition $GPSAltitude)
```

## Known use cases

[pxquim.pt](http://pxquim.pt/) uses Photos and the plug-in Sub-parts to publish 600 photo galleries with 40,000 photos. Photos keeps the high-resolution photos separate from the site articles.

[pxquim.com](http://pxquim.com/) uses sub-parts to cover conferences, where it makes sense to have a sub-part for each speaker.

## Alternatives

Gallery
:	Galleries are distinct entities, without the organizational capabilities of articles. Photos must be resized separately, and must be kept with the source of the blog. Gallery was the initial inspiration for Photos.

Image_process
:	Resize and process images in the article body in a more flexible way (based on the CSS class of the image), but without the ability to create galleries. The source photos must be kept with the source of the blog.

Contributing
------------

Contributions are welcome and much appreciated. Every little bit helps. You can contribute by improving the documentation, adding missing features, and fixing bugs. You can also help out by reviewing and commenting on [existing issues][].

 To start contributing to this plugin, review the [Contributing to Pelican][] documentation, beginning with the **Contributing Code** section.

[existing issues]: https://github.com/pelican-plugins/photos/issues
[Contributing to Pelican]: https://docs.getpelican.com/en/latest/contribute.html

License
-------

This project is licensed under the AGPL-3.0 license.
