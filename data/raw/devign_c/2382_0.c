static int draw_glyphs(DrawTextContext *s, AVFrame *frame,

                       int width, int height,

                       const uint8_t rgbcolor[4], const uint8_t yuvcolor[4],

                       int x, int y)

{

    char *text = HAVE_LOCALTIME_R ? s->expanded_text : s->text;

    uint32_t code = 0;

    int i;

    uint8_t *p;

    Glyph *glyph = NULL;



    for (i = 0, p = text; *p; i++) {

        Glyph dummy = { 0 };

        GET_UTF8(code, *p++, continue;);



        /* skip new line chars, just go to new line */

        if (code == '\n' || code == '\r' || code == '\t')

            continue;



        dummy.code = code;

        glyph = av_tree_find(s->glyphs, &dummy, (void *)glyph_cmp, NULL);



        if (glyph->bitmap.pixel_mode != FT_PIXEL_MODE_MONO &&

            glyph->bitmap.pixel_mode != FT_PIXEL_MODE_GRAY)

            return AVERROR(EINVAL);



        if (s->is_packed_rgb) {

            draw_glyph_rgb(frame, &glyph->bitmap,

                           s->positions[i].x+x, s->positions[i].y+y, width, height,

                           s->pixel_step[0], rgbcolor, s->rgba_map, s->alpha);

        } else {

            draw_glyph_yuv(frame, &glyph->bitmap,

                           s->positions[i].x+x, s->positions[i].y+y, width, height,

                           yuvcolor, s->hsub, s->vsub, s->alpha);

        }

    }



    return 0;

}
