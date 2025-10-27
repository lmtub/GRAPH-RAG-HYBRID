static void id3v2_read_ttag(AVFormatContext *s, int taglen, char *dst, int dstlen)

{

    char *q;

    int len;



    if(dstlen > 0)

        dst[0]= 0;

    if(taglen < 1)

        return;



    taglen--; /* account for encoding type byte */

    dstlen--; /* Leave space for zero terminator */



    switch(get_byte(s->pb)) { /* encoding type */



    case 0:  /* ISO-8859-1 (0 - 255 maps directly into unicode) */

        q = dst;

        while(taglen--) {

            uint8_t tmp;

            PUT_UTF8(get_byte(s->pb), tmp, if (q - dst < dstlen - 1) *q++ = tmp;)

        }

        *q = '\0';

        break;



    case 3:  /* UTF-8 */

        len = FFMIN(taglen, dstlen);

        get_buffer(s->pb, dst, len);

        dst[len] = 0;

        break;

    }

}
