static void expand_rle_row(unsigned char *optr, unsigned char *iptr, 

        int chan_offset, int pixelstride)

{

    unsigned char pixel, count;

 

#ifndef WORDS_BIGENDIAN

    /* rgba -> bgra for rgba32 on little endian cpus */

    if (pixelstride == 4 && chan_offset != 3) {

       chan_offset = 2 - chan_offset;

    }

#endif

        

    optr += chan_offset;



    while (1) {

        pixel = *iptr++;



        if (!(count = (pixel & 0x7f))) {

            return;

        }

        if (pixel & 0x80) {

            while (count--) {

                *optr = *iptr;

                optr += pixelstride;

                iptr++;

            }

        } else {

            pixel = *iptr++;



            while (count--) {

                *optr = pixel;

                optr += pixelstride;

            }

        }

    }

}
