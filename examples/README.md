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

Profile the code
----------------

Running the following command will use the Python [cProfile](https://docs.python.org/3/library/profile.html#module-cProfile) module to analyse/profile the Python code and the function calls.

```shell
make html-profile
```

By default it will create a file called ```pelican.pstats```. To dig deeper into the pstats file you can use some 3party tools/viwers.

- https://github.com/jiffyclub/snakeviz/
- https://github.com/jrfonseca/gprof2dot

License
-------

Code from the following projects has been included in this example

- https://zeptojs.com/ - MIT
- https://dimsemenov.com/plugins/magnific-popup/ - MIT
