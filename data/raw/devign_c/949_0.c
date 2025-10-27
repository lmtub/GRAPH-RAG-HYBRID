static int draw_text(AVFilterContext *ctx, AVFrame *frame,

                     int width, int height)

{

    DrawTextContext *s = ctx->priv;

    AVFilterLink *inlink = ctx->inputs[0];



    uint32_t code = 0, prev_code = 0;

    int x = 0, y = 0, i = 0, ret;

    int max_text_line_w = 0, len;

    int box_w, box_h;

    char *text;

    uint8_t *p;

    int y_min = 32000, y_max = -32000;

    int x_min = 32000, x_max = -32000;

    FT_Vector delta;

    Glyph *glyph = NULL, *prev_glyph = NULL;

    Glyph dummy = { 0 };



    time_t now = time(0);

    struct tm ltime;

    AVBPrint *bp = &s->expanded_text;



    FFDrawColor fontcolor;

    FFDrawColor shadowcolor;

    FFDrawColor bordercolor;

    FFDrawColor boxcolor;



    av_bprint_clear(bp);



    if(s->basetime != AV_NOPTS_VALUE)

        now= frame->pts*av_q2d(ctx->inputs[0]->time_base) + s->basetime/1000000;



    switch (s->exp_mode) {

    case EXP_NONE:

        av_bprintf(bp, "%s", s->text);

        break;

    case EXP_NORMAL:

        if ((ret = expand_text(ctx, s->text, &s->expanded_text)) < 0)

            return ret;

        break;

    case EXP_STRFTIME:

        localtime_r(&now, &ltime);

        av_bprint_strftime(bp, s->text, &ltime);

        break;

    }



    if (s->tc_opt_string) {

        char tcbuf[AV_TIMECODE_STR_SIZE];

        av_timecode_make_string(&s->tc, tcbuf, inlink->frame_count);

        av_bprint_clear(bp);

        av_bprintf(bp, "%s%s", s->text, tcbuf);

    }



    if (!av_bprint_is_complete(bp))

        return AVERROR(ENOMEM);

    text = s->expanded_text.str;

    if ((len = s->expanded_text.len) > s->nb_positions) {

        if (!(s->positions =

              av_realloc(s->positions, len*sizeof(*s->positions))))

            return AVERROR(ENOMEM);

        s->nb_positions = len;

    }



    if (s->fontcolor_expr[0]) {

        /* If expression is set, evaluate and replace the static value */

        av_bprint_clear(&s->expanded_fontcolor);

        if ((ret = expand_text(ctx, s->fontcolor_expr, &s->expanded_fontcolor)) < 0)

            return ret;

        if (!av_bprint_is_complete(&s->expanded_fontcolor))

            return AVERROR(ENOMEM);

        av_log(s, AV_LOG_DEBUG, "Evaluated fontcolor is '%s'\n", s->expanded_fontcolor.str);

        ret = av_parse_color(s->fontcolor.rgba, s->expanded_fontcolor.str, -1, s);

        if (ret)

            return ret;

        ff_draw_color(&s->dc, &s->fontcolor, s->fontcolor.rgba);

    }



    x = 0;

    y = 0;



    /* load and cache glyphs */

    for (i = 0, p = text; *p; i++) {

        GET_UTF8(code, *p++, continue;);



        /* get glyph */

        dummy.code = code;

        glyph = av_tree_find(s->glyphs, &dummy, glyph_cmp, NULL);

        if (!glyph) {

            load_glyph(ctx, &glyph, code);

        }



        y_min = FFMIN(glyph->bbox.yMin, y_min);

        y_max = FFMAX(glyph->bbox.yMax, y_max);

        x_min = FFMIN(glyph->bbox.xMin, x_min);

        x_max = FFMAX(glyph->bbox.xMax, x_max);

    }

    s->max_glyph_h = y_max - y_min;

    s->max_glyph_w = x_max - x_min;



    /* compute and save position for each glyph */

    glyph = NULL;

    for (i = 0, p = text; *p; i++) {

        GET_UTF8(code, *p++, continue;);



        /* skip the \n in the sequence \r\n */

        if (prev_code == '\r' && code == '\n')

            continue;



        prev_code = code;

        if (is_newline(code)) {



            max_text_line_w = FFMAX(max_text_line_w, x);

            y += s->max_glyph_h;

            x = 0;

            continue;

        }



        /* get glyph */

        prev_glyph = glyph;

        dummy.code = code;

        glyph = av_tree_find(s->glyphs, &dummy, glyph_cmp, NULL);



        /* kerning */

        if (s->use_kerning && prev_glyph && glyph->code) {

            FT_Get_Kerning(s->face, prev_glyph->code, glyph->code,

                           ft_kerning_default, &delta);

            x += delta.x >> 6;

        }



        /* save position */

        s->positions[i].x = x + glyph->bitmap_left;

        s->positions[i].y = y - glyph->bitmap_top + y_max;

        if (code == '\t') x  = (x / s->tabsize + 1)*s->tabsize;

        else              x += glyph->advance;

    }



    max_text_line_w = FFMAX(x, max_text_line_w);



    s->var_values[VAR_TW] = s->var_values[VAR_TEXT_W] = max_text_line_w;

    s->var_values[VAR_TH] = s->var_values[VAR_TEXT_H] = y + s->max_glyph_h;



    s->var_values[VAR_MAX_GLYPH_W] = s->max_glyph_w;

    s->var_values[VAR_MAX_GLYPH_H] = s->max_glyph_h;

    s->var_values[VAR_MAX_GLYPH_A] = s->var_values[VAR_ASCENT ] = y_max;

    s->var_values[VAR_MAX_GLYPH_D] = s->var_values[VAR_DESCENT] = y_min;



    s->var_values[VAR_LINE_H] = s->var_values[VAR_LH] = s->max_glyph_h;



    s->x = s->var_values[VAR_X] = av_expr_eval(s->x_pexpr, s->var_values, &s->prng);

    s->y = s->var_values[VAR_Y] = av_expr_eval(s->y_pexpr, s->var_values, &s->prng);

    s->x = s->var_values[VAR_X] = av_expr_eval(s->x_pexpr, s->var_values, &s->prng);



    update_alpha(s);

    update_color_with_alpha(s, &fontcolor  , s->fontcolor  );

    update_color_with_alpha(s, &shadowcolor, s->shadowcolor);

    update_color_with_alpha(s, &bordercolor, s->bordercolor);

    update_color_with_alpha(s, &boxcolor   , s->boxcolor   );



    box_w = FFMIN(width - 1 , max_text_line_w);

    box_h = FFMIN(height - 1, y + s->max_glyph_h);



    /* draw box */

    if (s->draw_box)

        ff_blend_rectangle(&s->dc, &boxcolor,

                           frame->data, frame->linesize, width, height,

                           s->x - s->boxborderw, s->y - s->boxborderw,

                           box_w + s->boxborderw * 2, box_h + s->boxborderw * 2);



    if (s->shadowx || s->shadowy) {

        if ((ret = draw_glyphs(s, frame, width, height,

                               &shadowcolor, s->shadowx, s->shadowy, 0)) < 0)

            return ret;

    }



    if (s->borderw) {

        if ((ret = draw_glyphs(s, frame, width, height,

                               &bordercolor, 0, 0, s->borderw)) < 0)

            return ret;

    }

    if ((ret = draw_glyphs(s, frame, width, height,

                           &fontcolor, 0, 0, 0)) < 0)

        return ret;



    return 0;

}
