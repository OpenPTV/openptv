#include <stdio.h>
#include <tiffio.h>

int main() {
    TIFF* tif = TIFFOpen("your_image.tif", "r");

    if (tif) {
        uint32_t width, height;
        uint32_t* raster;

        // Check if it's a grayscale image.
        if (TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel) && samplesPerPixel == 1) {
            TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
            TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);

            raster = (uint32_t*) _TIFFmalloc(width * height * sizeof(uint32_t));

            if (raster != NULL) {
                if (TIFFReadScanline(tif, raster, 0, 0)) {
                    // Successfully read the grayscale image, 'raster' contains the pixel data.
                    // You can access individual pixels using raster[x + y * width].

                    // Now you can work with the pixel data as needed.

                    _TIFFfree(raster);
                }
            }
        } else {
            printf("The image is not grayscale.\n");
        }

        TIFFClose(tif);
    }

    return 0;
}
