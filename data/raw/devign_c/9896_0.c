static void cin_decode_lzss(const unsigned char *src, int src_size, unsigned char *dst, int dst_size)

{

    uint16_t cmd;

    int i, sz, offset, code;

    unsigned char *dst_end = dst + dst_size;

    const unsigned char *src_end = src + src_size;



    while (src < src_end && dst < dst_end) {

        code = *src++;

        for (i = 0; i < 8 && src < src_end && dst < dst_end; ++i) {

            if (code & (1 << i)) {

                *dst++ = *src++;

            } else {

                cmd = AV_RL16(src); src += 2;

                offset = cmd >> 4;

                sz = (cmd & 0xF) + 2;

                /* don't use memcpy/memmove here as the decoding routine (ab)uses */

                /* buffer overlappings to repeat bytes in the destination */

                sz = FFMIN(sz, dst_end - dst);

                while (sz--) {

                    *dst = *(dst - offset - 1);

                    ++dst;

                }

            }

        }

    }

}
