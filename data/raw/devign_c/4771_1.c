static int expand_rle_row(const uint8_t *in_buf, const uint8_t* in_end,

            unsigned char *out_buf, uint8_t* out_end, int pixelstride)

{

    unsigned char pixel, count;

    unsigned char *orig = out_buf;



    while (1) {

        if(in_buf + 1 > in_end) return -1;

        pixel = bytestream_get_byte(&in_buf);

        if (!(count = (pixel & 0x7f))) {

            return (out_buf - orig) / pixelstride;

        }



        /* Check for buffer overflow. */

        if(out_buf + pixelstride * count >= out_end) return -1;



        if (pixel & 0x80) {

            while (count--) {

                *out_buf = bytestream_get_byte(&in_buf);

                out_buf += pixelstride;

            }

        } else {

            pixel = bytestream_get_byte(&in_buf);



            while (count--) {

                *out_buf = pixel;

                out_buf += pixelstride;

            }

        }

    }

}
