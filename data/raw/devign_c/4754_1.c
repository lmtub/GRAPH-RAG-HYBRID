static void qtrle_decode_16bpp(QtrleContext *s, int stream_ptr, int row_ptr, int lines_to_change)

{

    int rle_code;

    int pixel_ptr;

    int row_inc = s->frame.linesize[0];

    unsigned short rgb16;

    unsigned char *rgb = s->frame.data[0];

    int pixel_limit = s->frame.linesize[0] * s->avctx->height;



    while (lines_to_change--) {

        CHECK_STREAM_PTR(2);

        pixel_ptr = row_ptr + (s->buf[stream_ptr++] - 1) * 2;



        while ((rle_code = (signed char)s->buf[stream_ptr++]) != -1) {

            if (rle_code == 0) {

                /* there's another skip code in the stream */

                CHECK_STREAM_PTR(1);

                pixel_ptr += (s->buf[stream_ptr++] - 1) * 2;

                CHECK_PIXEL_PTR(0);  /* make sure pixel_ptr is positive */

            } else if (rle_code < 0) {

                /* decode the run length code */

                rle_code = -rle_code;

                CHECK_STREAM_PTR(2);

                rgb16 = AV_RB16(&s->buf[stream_ptr]);

                stream_ptr += 2;



                CHECK_PIXEL_PTR(rle_code * 2);



                while (rle_code--) {

                    *(unsigned short *)(&rgb[pixel_ptr]) = rgb16;

                    pixel_ptr += 2;

                }

            } else {

                CHECK_STREAM_PTR(rle_code * 2);

                CHECK_PIXEL_PTR(rle_code * 2);



                /* copy pixels directly to output */

                while (rle_code--) {

                    rgb16 = AV_RB16(&s->buf[stream_ptr]);

                    stream_ptr += 2;

                    *(unsigned short *)(&rgb[pixel_ptr]) = rgb16;

                    pixel_ptr += 2;

                }

            }

        }

        row_ptr += row_inc;

    }

}
