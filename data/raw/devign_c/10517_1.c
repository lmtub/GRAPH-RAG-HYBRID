static int png_decode_idat(PNGDecContext *s, int length)

{

    int ret;

    s->zstream.avail_in = FFMIN(length, bytestream2_get_bytes_left(&s->gb));

    s->zstream.next_in  = (unsigned char *)s->gb.buffer;

    bytestream2_skip(&s->gb, length);



    /* decode one line if possible */

    while (s->zstream.avail_in > 0) {

        ret = inflate(&s->zstream, Z_PARTIAL_FLUSH);

        if (ret != Z_OK && ret != Z_STREAM_END) {

            av_log(s->avctx, AV_LOG_ERROR, "inflate returned error %d\n", ret);

            return AVERROR_EXTERNAL;

        }

        if (s->zstream.avail_out == 0) {

            if (!(s->state & PNG_ALLIMAGE)) {

                png_handle_row(s);

            }

            s->zstream.avail_out = s->crow_size;

            s->zstream.next_out  = s->crow_buf;

        }

        if (ret == Z_STREAM_END && s->zstream.avail_in > 0) {

            av_log(NULL, AV_LOG_WARNING,

                   "%d undecompressed bytes left in buffer\n", s->zstream.avail_in);

            return 0;

        }

    }

    return 0;

}
