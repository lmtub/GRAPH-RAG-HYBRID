static int delta_decode(int8_t *dst, const uint8_t *src, int src_size,

                        int8_t val, const int8_t *table)

{

    int n = src_size;

    int8_t *dst0 = dst;



    while (n--) {

        uint8_t d = *src++;

        val = av_clip(val + table[d & 0x0f], -127, 128);

        *dst++ = val;

        val = av_clip(val + table[d >> 4]  , -127, 128);

        *dst++ = val;

    }



    return dst-dst0;

}
