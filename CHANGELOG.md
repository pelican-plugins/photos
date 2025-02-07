CHANGELOG
=========

1.6.2 - 2025-02-07
------------------

* Convert p image to remove transparency
* Improve support for i18n_subsites

Contributed by [PhiBo](https://github.com/phibos) via [PR #131](https://github.com/pelican-plugins/photos/pull/131/)


1.6.1 - 2025-01-30
------------------

* Update dependencies

Contributed by [PhiBo](https://github.com/phibos) via [PR #128](https://github.com/pelican-plugins/photos/pull/128/)


1.6.0 - 2024-03-17
------------------

* Fix issue with Pillow 10.1+
* Fix issue with font size to small
* Add SITEURL to template context
* Add more examples
* Update CI pipeline

1.5.0 - 2023-10-30
------------------

* Improve speed by using a global process pool
* Fix exclude list format issue
* Catch exceptions while reading or writing EXIF data
* Handle photos for which GPS data is not available
* Handle missing galleries
* Post-process JPEG images with RGB converter

1.4.0 - 2022-09-08
------------------

- Add config parameter to generate addtional images
- Add optional to calculate the average color for an image
- Add Profile support
- Rewrite processing
- Find inline pattern and replace with image/gallery
- New operation convert_mode_p
- Add convert and quantize operations
- Add support for named images in metadata
- Process images from all pages and articles (Fixes: #36)
- Raise exception if unable to detect mime type
- Skip subdirectories in galleries
- Raise exception if file is not an image
- Improve info file handling
- Use icc profile from source image
- Handle EXIF and Rotate images (Fixes: #49)
- Improve URL generation with SITEURL

Contributed by [PhiBo](https://github.com/phibos) via [PR #61](https://github.com/pelican-plugins/photos/pull/61/)


1.3.0 - 2022-01-07
------------------

- Add default processing information and pass them to the Pillow writer
- Fix issues with multiprocessing
- Add option to detect number of CPUs
- Add new class to hold information about the source image
- Add caching for exif, notes and exclude lists
- Add high and width information from the result image
- Update signal and config handling
- Add optional api documentation
- Add more docstrings
- Add an initial version of an example blog to show how to use the plugin

Contributed by [PhiBo](https://github.com/phibos) [PR #25](https://github.com/pelican-plugins/photos/pull/25/)


1.2.0 - 2022-01-01
------------------

- Change code to use classes instead of tuples to hold image information
- Add support to select image compression by source type. (jpeg -> webp, gif -> png, ...)
- Add support for HTML srcset

Contributed by [PhiBo](https://github.com/phibos) [PR #16](https://github.com/pelican-plugins/photos/pull/16/)


1.1.0 - 2021-12-27
------------------

Add support for inline galleries.

Contributed by [PhiBo](https://github.com/phibos) [PR #13](https://github.com/pelican-plugins/photos/pull/13/)


1.0.1 - 2021-08-26
------------------

Fix minor issues with some static files not included in the package.

Contributed by [PhiBo](https://github.com/phibos) [PR #7](https://github.com/pelican-plugins/photos/pull/7/)


1.0.0 - 2021-08-13
------------------

Initial release as namespace plugin
