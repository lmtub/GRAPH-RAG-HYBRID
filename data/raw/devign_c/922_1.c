static void cin_decode_rle(const unsigned char *src, int src_size, unsigned char *dst, int dst_size)

{

    int len, code;

    unsigned char *dst_end = dst + dst_size;

    const unsigned char *src_end = src + src_size;



    while (src < src_end && dst < dst_end) {

        code = *src++;

        if (code & 0x80) {

            len = code - 0x7F;

            memset(dst, *src++, FFMIN(len, dst_end - dst));

        } else {

            len = code + 1;

            memcpy(dst, src, FFMIN(len, dst_end - dst));

            src += len;

        }

        dst += len;

    }

}
