How to use photos plugin for for Pelican
========================================

This is an example how to use the photos plugin for pelican. We have included the default `notmyidea` theme and striped some files not required to show how to modify the template to include the generated photos. This is far from perfect but we tried to keep it as simple as possible. Just look for HTML comments with `PHOTO plugin:`.

Install
-------

```shell
python -v venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Bootstrap
---------

We do not include the example images in the repository. But a small script is provided to download the required images from Wikipedia.

```shell
./download_images.sh
```

Build make
----------

If everything has been successful you can start the build process.

```shell
make devserver
```

License
-------

Code from the following projects has been included in this example

- https://zeptojs.com/ - MIT
- https://dimsemenov.com/plugins/magnific-popup/ - MIT
