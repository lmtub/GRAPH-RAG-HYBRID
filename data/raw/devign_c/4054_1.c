static const char *srt_to_ass(AVCodecContext *avctx, char *out, char *out_end,

                              const char *in, int x1, int y1, int x2, int y2)

{

    char c, *param, buffer[128], tmp[128];

    int len, tag_close, sptr = 1, line_start = 1, an = 0, end = 0;

    SrtStack stack[16];



    stack[0].tag[0] = 0;

    strcpy(stack[0].param[PARAM_SIZE],  "{\\fs}");

    strcpy(stack[0].param[PARAM_COLOR], "{\\c}");

    strcpy(stack[0].param[PARAM_FACE],  "{\\fn}");



    if (x1 >= 0 && y1 >= 0) {

        if (x2 >= 0 && y2 >= 0 && (x2 != x1 || y2 != y1))

            out += snprintf(out, out_end-out,

                            "{\\an1}{\\move(%d,%d,%d,%d)}", x1, y1, x2, y2);

        else

            out += snprintf(out, out_end-out, "{\\an1}{\\pos(%d,%d)}", x1, y1);

    }



    for (; out < out_end && !end && *in; in++) {

        switch (*in) {

        case '\r':

            break;

        case '\n':

            if (line_start) {

                end = 1;

                break;

            }

            while (out[-1] == ' ')

                out--;

            out += snprintf(out, out_end-out, "\\N");

            line_start = 1;

            break;

        case ' ':

            if (!line_start)

                *out++ = *in;

            break;

        case '{':    /* skip all {\xxx} substrings except for {\an%d}

                        and all microdvd like styles such as {Y:xxx} */

            an += sscanf(in, "{\\an%*1u}%c", &c) == 1;

            if ((an != 1 && sscanf(in, "{\\%*[^}]}%n%c", &len, &c) > 0) ||

                sscanf(in, "{%*1[CcFfoPSsYy]:%*[^}]}%n%c", &len, &c) > 0) {

                in += len - 1;

            } else

                *out++ = *in;

            break;

        case '<':

            tag_close = in[1] == '/';

            if (sscanf(in+tag_close+1, "%127[^>]>%n%c", buffer, &len,&c) >= 2) {

                if ((param = strchr(buffer, ' ')))

                    *param++ = 0;

                if ((!tag_close && sptr < FF_ARRAY_ELEMS(stack)) ||

                    ( tag_close && sptr > 0 && !strcmp(stack[sptr-1].tag, buffer))) {

                    int i, j, unknown = 0;

                    in += len + tag_close;

                    if (!tag_close)

                        memset(stack+sptr, 0, sizeof(*stack));

                    if (!strcmp(buffer, "font")) {

                        if (tag_close) {

                            for (i=PARAM_NUMBER-1; i>=0; i--)

                                if (stack[sptr-1].param[i][0])

                                    for (j=sptr-2; j>=0; j--)

                                        if (stack[j].param[i][0]) {

                                            out += snprintf(out, out_end-out,

                                                            stack[j].param[i]);

                                            break;

                                        }

                        } else {

                            while (param) {

                                if (!strncmp(param, "size=", 5)) {

                                    unsigned font_size;

                                    param += 5 + (param[5] == '"');

                                    if (sscanf(param, "%u", &font_size) == 1) {

                                        snprintf(stack[sptr].param[PARAM_SIZE],

                                             sizeof(stack[0].param[PARAM_SIZE]),

                                             "{\\fs%u}", font_size);

                                    }

                                } else if (!strncmp(param, "color=", 6)) {

                                    param += 6 + (param[6] == '"');

                                    snprintf(stack[sptr].param[PARAM_COLOR],

                                         sizeof(stack[0].param[PARAM_COLOR]),

                                         "{\\c&H%X&}",

                                         html_color_parse(avctx, param));

                                } else if (!strncmp(param, "face=", 5)) {

                                    param += 5 + (param[5] == '"');

                                    len = strcspn(param,

                                                  param[-1] == '"' ? "\"" :" ");

                                    av_strlcpy(tmp, param,

                                               FFMIN(sizeof(tmp), len+1));

                                    param += len;

                                    snprintf(stack[sptr].param[PARAM_FACE],

                                             sizeof(stack[0].param[PARAM_FACE]),

                                             "{\\fn%s}", tmp);

                                }

                                if ((param = strchr(param, ' ')))

                                    param++;

                            }

                            for (i=0; i<PARAM_NUMBER; i++)

                                if (stack[sptr].param[i][0])

                                    out += snprintf(out, out_end-out,

                                                    stack[sptr].param[i]);

                        }

                    } else if (!buffer[1] && strspn(buffer, "bisu") == 1) {

                        out += snprintf(out, out_end-out,

                                        "{\\%c%d}", buffer[0], !tag_close);

                    } else {

                        unknown = 1;

                        snprintf(tmp, sizeof(tmp), "</%s>", buffer);

                    }

                    if (tag_close) {

                        sptr--;

                    } else if (unknown && !strstr(in, tmp)) {

                        in -= len + tag_close;

                        *out++ = *in;

                    } else

                        av_strlcpy(stack[sptr++].tag, buffer,

                                   sizeof(stack[0].tag));

                    break;

                }

            }

        default:

            *out++ = *in;

            break;

        }

        if (*in != ' ' && *in != '\r' && *in != '\n')

            line_start = 0;

    }



    out = FFMIN(out, out_end-3);

    while (!strncmp(out-2, "\\N", 2))

        out -= 2;

    while (out[-1] == ' ')

        out--;

    out += snprintf(out, out_end-out, "\r\n");

    return in;

}
