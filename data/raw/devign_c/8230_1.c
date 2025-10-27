int ff_mjpeg_find_marker(MJpegDecodeContext *s,

                         const uint8_t **buf_ptr, const uint8_t *buf_end,

                         const uint8_t **unescaped_buf_ptr,

                         int *unescaped_buf_size)

{

    int start_code;

    start_code = find_marker(buf_ptr, buf_end);



    av_fast_padded_malloc(&s->buffer, &s->buffer_size, buf_end - *buf_ptr);

    if (!s->buffer)

        return AVERROR(ENOMEM);



    /* unescape buffer of SOS, use special treatment for JPEG-LS */

    if (start_code == SOS && !s->ls) {

        const uint8_t *src = *buf_ptr;

        uint8_t *dst = s->buffer;



        while (src < buf_end) {

            uint8_t x = *(src++);



            *(dst++) = x;

            if (s->avctx->codec_id != AV_CODEC_ID_THP) {

                if (x == 0xff) {

                    while (src < buf_end && x == 0xff)

                        x = *(src++);



                    if (x >= 0xd0 && x <= 0xd7)

                        *(dst++) = x;

                    else if (x)

                        break;

                }

            }

        }

        *unescaped_buf_ptr  = s->buffer;

        *unescaped_buf_size = dst - s->buffer;

        memset(s->buffer + *unescaped_buf_size, 0,

               AV_INPUT_BUFFER_PADDING_SIZE);



        av_log(s->avctx, AV_LOG_DEBUG, "escaping removed %"PTRDIFF_SPECIFIER" bytes\n",

               (buf_end - *buf_ptr) - (dst - s->buffer));

    } else if (start_code == SOS && s->ls) {

        const uint8_t *src = *buf_ptr;

        uint8_t *dst  = s->buffer;

        int bit_count = 0;

        int t = 0, b = 0;

        PutBitContext pb;



        /* find marker */

        while (src + t < buf_end) {

            uint8_t x = src[t++];

            if (x == 0xff) {

                while ((src + t < buf_end) && x == 0xff)

                    x = src[t++];

                if (x & 0x80) {

                    t -= FFMIN(2, t);

                    break;

                }

            }

        }

        bit_count = t * 8;

        init_put_bits(&pb, dst, t);



        /* unescape bitstream */

        while (b < t) {

            uint8_t x = src[b++];

            put_bits(&pb, 8, x);

            if (x == 0xFF) {

                x = src[b++];

                if (x & 0x80) {

                    av_log(s->avctx, AV_LOG_WARNING, "Invalid escape sequence\n");

                    x &= 0x7f;

                }

                put_bits(&pb, 7, x);

                bit_count--;

            }

        }

        flush_put_bits(&pb);



        *unescaped_buf_ptr  = dst;

        *unescaped_buf_size = (bit_count + 7) >> 3;

        memset(s->buffer + *unescaped_buf_size, 0,

               AV_INPUT_BUFFER_PADDING_SIZE);

    } else {

        *unescaped_buf_ptr  = *buf_ptr;

        *unescaped_buf_size = buf_end - *buf_ptr;

    }



    return start_code;

}
