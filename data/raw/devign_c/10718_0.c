static int delta_decode(uint8_t *dst, const uint8_t *src, int src_size,

                         unsigned val, const int8_t *table)

{

    uint8_t *dst0 = dst;



    while (src_size--) {

        uint8_t d = *src++;

        val = av_clip_uint8(val + table[d & 0xF]);

        *dst++ = val;

        val = av_clip_uint8(val + table[d >> 4]);

        *dst++ = val;

    }



    return dst-dst0;

}
