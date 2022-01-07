CHANGELOG
=========

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
