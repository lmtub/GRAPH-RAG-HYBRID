static int decode_byterun(uint8_t *dst, int dst_size,

                          const uint8_t *buf, const uint8_t *const buf_end)

{

    const uint8_t *const buf_start = buf;

    unsigned x;

    for (x = 0; x < dst_size && buf < buf_end;) {

        unsigned length;

        const int8_t value = *buf++;

        if (value >= 0) {

            length = value + 1;

            memcpy(dst + x, buf, FFMIN3(length, dst_size - x, buf_end - buf));

            buf += length;

        } else if (value > -128) {

            length = -value + 1;

            memset(dst + x, *buf++, FFMIN(length, dst_size - x));

        } else { // noop

            continue;

        }

        x += length;

    }

    if (x < dst_size) {

        av_log(NULL, AV_LOG_WARNING, "decode_byterun ended before plane size\n");

        memset(dst+x, 0, dst_size - x);

    }

    return buf - buf_start;

}
