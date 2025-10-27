char *av_base64_encode(uint8_t * src, int len)

{

    static const char b64[] =

        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    char *ret, *dst;

    unsigned i_bits = 0;

    int i_shift = 0;

    int bytes_remaining = len;



    if (len < UINT_MAX / 4) {

        ret = dst = av_malloc(len * 4 / 3 + 12);

    } else

        return NULL;



    if (len) {                  // special edge case, what should we really do here?

        while (bytes_remaining) {

            i_bits = (i_bits << 8) + *src++;

            bytes_remaining--;

            i_shift += 8;



            do {

                *dst++ = b64[(i_bits << 6 >> i_shift) & 0x3f];

                i_shift -= 6;

            } while (i_shift > 6 || (bytes_remaining == 0 && i_shift > 0));

        }

        while ((dst - ret) & 3)

            *dst++ = '=';

    }

    *dst = '\0';



    return ret;

}
