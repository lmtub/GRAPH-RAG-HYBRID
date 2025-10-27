static void qtrle_decode_32bpp(QtrleContext *s, int stream_ptr, int row_ptr, int lines_to_change)

{

    int rle_code;

    int pixel_ptr;

    int row_inc = s->frame.linesize[0];

    unsigned char a, r, g, b;

    unsigned int argb;

    unsigned char *rgb = s->frame.data[0];

    int pixel_limit = s->frame.linesize[0] * s->avctx->height;



    while (lines_to_change--) {

        CHECK_STREAM_PTR(2);

        pixel_ptr = row_ptr + (s->buf[stream_ptr++] - 1) * 4;




        while ((rle_code = (signed char)s->buf[stream_ptr++]) != -1) {

            if (rle_code == 0) {

                /* there's another skip code in the stream */

                CHECK_STREAM_PTR(1);

                pixel_ptr += (s->buf[stream_ptr++] - 1) * 4;


            } else if (rle_code < 0) {

                /* decode the run length code */

                rle_code = -rle_code;

                CHECK_STREAM_PTR(4);

                a = s->buf[stream_ptr++];

                r = s->buf[stream_ptr++];

                g = s->buf[stream_ptr++];

                b = s->buf[stream_ptr++];

                argb = (a << 24) | (r << 16) | (g << 8) | (b << 0);



                CHECK_PIXEL_PTR(rle_code * 4);



                while (rle_code--) {

                    *(unsigned int *)(&rgb[pixel_ptr]) = argb;

                    pixel_ptr += 4;

                }

            } else {

                CHECK_STREAM_PTR(rle_code * 4);

                CHECK_PIXEL_PTR(rle_code * 4);



                /* copy pixels directly to output */

                while (rle_code--) {

                    a = s->buf[stream_ptr++];

                    r = s->buf[stream_ptr++];

                    g = s->buf[stream_ptr++];

                    b = s->buf[stream_ptr++];

                    argb = (a << 24) | (r << 16) | (g << 8) | (b << 0);

                    *(unsigned int *)(&rgb[pixel_ptr]) = argb;

                    pixel_ptr += 4;

                }

            }

        }

        row_ptr += row_inc;

    }

}