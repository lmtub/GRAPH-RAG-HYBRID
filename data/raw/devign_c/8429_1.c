int xbzrle_decode_buffer(uint8_t *src, int slen, uint8_t *dst, int dlen)

{

    int i = 0, d = 0;

    int ret;

    uint32_t count = 0;



    while (i < slen) {



        /* zrun */

        if ((slen - i) < 2) {

            return -1;

        }



        ret = uleb128_decode_small(src + i, &count);

        if (ret < 0 || (i && !count)) {

            return -1;

        }

        i += ret;

        d += count;



        /* overflow */

        if (d > dlen) {

            return -1;

        }



        /* nzrun */

        if ((slen - i) < 2) {

            return -1;

        }



        ret = uleb128_decode_small(src + i, &count);

        if (ret < 0 || !count) {

            return -1;

        }

        i += ret;



        /* overflow */

        if (d + count > dlen || i + count > slen) {

            return -1;

        }



        memcpy(dst + d, src + i, count);

        d += count;

        i += count;

    }



    return d;

}
