int ff_mov_iso639_to_lang(const char *lang, int mp4)

{

    int i, code = 0;



    /* old way, only for QT? */

    for (i = 0; !mp4 && i < FF_ARRAY_ELEMS(mov_mdhd_language_map); i++) {

        if (mov_mdhd_language_map[i] && !strcmp(lang, mov_mdhd_language_map[i]))

            return i;

    }

    /* XXX:can we do that in mov too? */

    if (!mp4)

        return 0;

    /* handle undefined as such */

    if (lang[0] == '\0')

        lang = "und";

    /* 5bit ascii */

    for (i = 0; i < 3; i++) {

        unsigned char c = (unsigned char)lang[i];

        if (c < 0x60)

            return 0;

        if (c > 0x60 + 0x1f)

            return 0;

        code <<= 5;

        code |= (c - 0x60);

    }

    return code;

}
