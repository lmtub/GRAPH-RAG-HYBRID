static int dtext_prepare_text(AVFilterContext *ctx)

{

    DrawTextContext *dtext = ctx->priv;

    uint32_t code = 0, prev_code = 0;

    int x = 0, y = 0, i = 0, ret;

    int text_height, baseline;

    char *text = dtext->text;

    uint8_t *p;

    int str_w = 0, len;

    int y_min = 32000, y_max = -32000;

    FT_Vector delta;

    Glyph *glyph = NULL, *prev_glyph = NULL;

    Glyph dummy = { 0 };

    int width  = ctx->inputs[0]->w;

    int height = ctx->inputs[0]->h;



#if HAVE_LOCALTIME_R

    time_t now = time(0);

    struct tm ltime;

    uint8_t *buf = dtext->expanded_text;

    int buf_size = dtext->expanded_text_size;



    if (!buf)

        buf_size = 2*strlen(dtext->text)+1;



    localtime_r(&now, &ltime);



    while ((buf = av_realloc(buf, buf_size))) {

        *buf = 1;

        if (strftime(buf, buf_size, dtext->text, &ltime) != 0 || *buf == 0)

            break;

        buf_size *= 2;

    }



    if (!buf)

        return AVERROR(ENOMEM);

    text = dtext->expanded_text = buf;

    dtext->expanded_text_size = buf_size;

#endif



    if ((len = strlen(text)) > dtext->nb_positions) {

        FT_Vector *p = av_realloc(dtext->positions,

                                  len * sizeof(*dtext->positions));

        if (!p) {

            av_freep(dtext->positions);

            dtext->nb_positions = 0;

            return AVERROR(ENOMEM);

        } else {

            dtext->positions = p;

            dtext->nb_positions = len;

        }

    }



    /* load and cache glyphs */

    for (i = 0, p = text; *p; i++) {

        GET_UTF8(code, *p++, continue;);



        /* get glyph */

        dummy.code = code;

        glyph = av_tree_find(dtext->glyphs, &dummy, glyph_cmp, NULL);

        if (!glyph)

            ret = load_glyph(ctx, &glyph, code);

        if (ret) return ret;



        y_min = FFMIN(glyph->bbox.yMin, y_min);

        y_max = FFMAX(glyph->bbox.yMax, y_max);

    }

    text_height = y_max - y_min;

    baseline    = y_max;



    /* compute and save position for each glyph */

    glyph = NULL;

    for (i = 0, p = text; *p; i++) {

        GET_UTF8(code, *p++, continue;);



        /* skip the \n in the sequence \r\n */

        if (prev_code == '\r' && code == '\n')

            continue;



        prev_code = code;

        if (is_newline(code)) {

            str_w = FFMAX(str_w, x - dtext->x);

            y += text_height;

            x = 0;

            continue;

        }



        /* get glyph */

        prev_glyph = glyph;

        dummy.code = code;

        glyph = av_tree_find(dtext->glyphs, &dummy, glyph_cmp, NULL);



        /* kerning */

        if (dtext->use_kerning && prev_glyph && glyph->code) {

            FT_Get_Kerning(dtext->face, prev_glyph->code, glyph->code,

                           ft_kerning_default, &delta);

            x += delta.x >> 6;

        }



        if (x + glyph->bbox.xMax >= width) {

            str_w = FFMAX(str_w, x);

            y += text_height;

            x = 0;

        }



        /* save position */

        dtext->positions[i].x = x + glyph->bitmap_left;

        dtext->positions[i].y = y - glyph->bitmap_top + baseline;

        if (code == '\t') x  = (x / dtext->tabsize + 1)*dtext->tabsize;

        else              x += glyph->advance;

    }



    str_w = FFMIN(width - 1, FFMAX(str_w, x));

    y     = FFMIN(y + text_height, height - 1);



    dtext->w = str_w;

    dtext->h = y;



    return 0;

}
