static void qtrle_decode_24bpp(QtrleContext *s, int stream_ptr, int row_ptr, int lines_to_change)

{

    int rle_code;

    int pixel_ptr;

    int row_inc = s->frame.linesize[0];

    unsigned char r, g, b;

    unsigned char *rgb = s->frame.data[0];

    int pixel_limit = s->frame.linesize[0] * s->avctx->height;



    while (lines_to_change--) {

        CHECK_STREAM_PTR(2);

        pixel_ptr = row_ptr + (s->buf[stream_ptr++] - 1) * 3;




        while ((rle_code = (signed char)s->buf[stream_ptr++]) != -1) {

            if (rle_code == 0) {

                /* there's another skip code in the stream */

                CHECK_STREAM_PTR(1);

                pixel_ptr += (s->buf[stream_ptr++] - 1) * 3;


            } else if (rle_code < 0) {

                /* decode the run length code */

                rle_code = -rle_code;

                CHECK_STREAM_PTR(3);

                r = s->buf[stream_ptr++];

                g = s->buf[stream_ptr++];

                b = s->buf[stream_ptr++];



                CHECK_PIXEL_PTR(rle_code * 3);



                while (rle_code--) {

                    rgb[pixel_ptr++] = r;

                    rgb[pixel_ptr++] = g;

                    rgb[pixel_ptr++] = b;

                }

            } else {

                CHECK_STREAM_PTR(rle_code * 3);

                CHECK_PIXEL_PTR(rle_code * 3);



                /* copy pixels directly to output */

                while (rle_code--) {

                    rgb[pixel_ptr++] = s->buf[stream_ptr++];

                    rgb[pixel_ptr++] = s->buf[stream_ptr++];

                    rgb[pixel_ptr++] = s->buf[stream_ptr++];

                }

            }

        }

        row_ptr += row_inc;

    }

}