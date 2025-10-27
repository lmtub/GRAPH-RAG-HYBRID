static int draw_glyphs(DrawTextContext *dtext, AVFilterBufferRef *picref,

                       int width, int height, const uint8_t rgbcolor[4], const uint8_t yuvcolor[4], int x, int y)

{

    char *text = dtext->text;

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

        glyph = av_tree_find(dtext->glyphs, &dummy, (void *)glyph_cmp, NULL);



        if (glyph->bitmap.pixel_mode != FT_PIXEL_MODE_MONO &&

            glyph->bitmap.pixel_mode != FT_PIXEL_MODE_GRAY)

            return AVERROR(EINVAL);



        if (dtext->is_packed_rgb) {

            draw_glyph_rgb(picref, &glyph->bitmap,

                           dtext->positions[i].x+x, dtext->positions[i].y+y, width, height,

                           dtext->pixel_step[0], rgbcolor, dtext->rgba_map);

        } else {

            draw_glyph_yuv(picref, &glyph->bitmap,

                           dtext->positions[i].x+x, dtext->positions[i].y+y, width, height,

                           yuvcolor, dtext->hsub, dtext->vsub);

        }

    }



    return 0;

}
