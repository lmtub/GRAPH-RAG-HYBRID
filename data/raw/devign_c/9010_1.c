static int xan_huffman_decode(uint8_t *dest, int dest_len,

                              const uint8_t *src, int src_len)

{

    uint8_t byte = *src++;

    uint8_t ival = byte + 0x16;

    const uint8_t * ptr = src + byte*2;

    int ptr_len = src_len - 1 - byte*2;

    uint8_t val = ival;

    uint8_t *dest_end = dest + dest_len;

    uint8_t *dest_start = dest;

    int ret;

    GetBitContext gb;



    if ((ret = init_get_bits8(&gb, ptr, ptr_len)) < 0)

        return ret;



    while (val != 0x16) {

        unsigned idx = val - 0x17 + get_bits1(&gb) * byte;

        if (idx >= 2 * byte)

            return AVERROR_INVALIDDATA;

        val = src[idx];



        if (val < 0x16) {

            if (dest >= dest_end)

                return dest_len;

            *dest++ = val;

            val = ival;

        }

    }



    return dest - dest_start;

}
