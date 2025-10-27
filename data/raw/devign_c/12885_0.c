static int escape(char **dst, const char *src, const char *special_chars,

                  enum EscapeMode mode)

{

    AVBPrint dstbuf;



    av_bprint_init(&dstbuf, 1, AV_BPRINT_SIZE_UNLIMITED);



    switch (mode) {

    case ESCAPE_MODE_FULL:

    case ESCAPE_MODE_LAZY:

        /* \-escape characters */



        if (mode == ESCAPE_MODE_LAZY && strchr(WHITESPACES, *src))

            av_bprintf(&dstbuf, "\\%c", *src++);



        for (; *src; src++) {

            if ((special_chars && strchr(special_chars, *src)) ||

                strchr("'\\", *src) ||

                (mode == ESCAPE_MODE_FULL && strchr(WHITESPACES, *src)))

                av_bprintf(&dstbuf, "\\%c", *src);

            else

                av_bprint_chars(&dstbuf, *src, 1);

        }



        if (mode == ESCAPE_MODE_LAZY && strchr(WHITESPACES, dstbuf.str[dstbuf.len-1])) {

            char c = dstbuf.str[dstbuf.len-1];

            dstbuf.str[dstbuf.len-1] = '\\';

            av_bprint_chars(&dstbuf, c, 1);

        }

        break;



    case ESCAPE_MODE_QUOTE:

        /* enclose between '' the string */

        av_bprint_chars(&dstbuf, '\'', 1);

        for (; *src; src++) {

            if (*src == '\'')

                av_bprintf(&dstbuf, "'\\''");

            else

                av_bprint_chars(&dstbuf, *src, 1);

        }

        av_bprint_chars(&dstbuf, '\'', 1);

        break;



    default:

        /* unknown escape mode */

        return AVERROR(EINVAL);

    }



    if (!av_bprint_is_complete(&dstbuf)) {

        av_bprint_finalize(&dstbuf, NULL);

        return AVERROR(ENOMEM);

    } else {

        av_bprint_finalize(&dstbuf, dst);

        return 0;

    }

}
