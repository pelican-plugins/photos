Release type: minor

- support for optional `inline_image.html` and `image_lightbox.html` templates
- `SITEURL` is now well substituted in all the `inline_xxx.html` templates
- the PHOTO_INLINE_GALLERY_PATTERN python regex also define an "options"
  dict that is transmitted to the inline_gallery.html template file
- added in the `README.md` file some documentation for these new features
- contributed by Pirogh <pierre.saramito@imag.fr>
