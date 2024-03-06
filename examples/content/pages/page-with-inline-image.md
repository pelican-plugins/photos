Title: Page with inline image
Summary: This is an example with an inline image.

## Use inline images

Just use the default syntax to add images in markdown.

```
![Alt text]({photo}example_gallery/Pelecanus_crispus_at_Beijing_Zoo_crop.JPG)
```

![Alt text]({photo}example_gallery/Pelecanus_crispus_at_Beijing_Zoo_crop.JPG)

## Use inline images with template

This will use the template specified with ```PHOTO_INLINE_IMAGE_TEMPLATE```.
The default is ```inline_image``` and will use the template file ```inline_image.html```.

### Minimal example

Just add a div-Tag.

```html
<div image="{photo}example_gallery/Pelecanus_crispus_at_Beijing_Zoo_crop.JPG"></div>
```

<div image="{photo}example_gallery/Pelecanus_crispus_at_Beijing_Zoo_crop.JPG"></div>

### Additional attributes

We can add additional attributes and use them in the template. In this example we use caption.

```html
<div image="{photo}example_gallery/Pelecanus_crispus_at_Beijing_Zoo_crop.JPG" caption="This is a custom caption"></div>
```

<div image="{photo}example_gallery/Pelecanus_crispus_at_Beijing_Zoo_crop.JPG" caption="This is a custom caption"></div>


Lightbox
--------

**You need JavaScript to show the image in an lightbox**

![Alt text]({lightbox}example_gallery/Pelecanus_crispus_at_Beijing_Zoo_crop.JPG)
