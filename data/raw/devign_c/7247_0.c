static void read_ttag(AVFormatContext *s, int taglen, const char *key)

{

    char *q, dst[512];

    int len, dstlen = sizeof(dst) - 1;

    unsigned genre;



    dst[0] = 0;

    if (taglen < 1)

        return;



    taglen--; /* account for encoding type byte */



    switch (get_byte(s->pb)) { /* encoding type */



    case 0:  /* ISO-8859-1 (0 - 255 maps directly into unicode) */

        q = dst;

        while (taglen--) {

            uint8_t tmp;

            PUT_UTF8(get_byte(s->pb), tmp, if (q - dst < dstlen - 1) *q++ = tmp;)

        }

        *q = '\0';

        break;



    case 3:  /* UTF-8 */

        len = FFMIN(taglen, dstlen - 1);

        get_buffer(s->pb, dst, len);

        dst[len] = 0;

        break;

    }



    if (!strcmp(key, "genre")

        && (sscanf(dst, "(%d)", &genre) == 1 || sscanf(dst, "%d", &genre) == 1)

        && genre <= ID3v1_GENRE_MAX)

        av_strlcpy(dst, ff_id3v1_genre_str[genre], sizeof(dst));



    if (*dst)

        av_metadata_set(&s->metadata, key, dst);

}
