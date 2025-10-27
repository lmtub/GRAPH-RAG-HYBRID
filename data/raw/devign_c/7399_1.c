static int load_glyph(AVFilterContext *ctx, Glyph **glyph_ptr, uint32_t code)

{

    DrawTextContext *s = ctx->priv;

    FT_BitmapGlyph bitmapglyph;

    Glyph *glyph;

    struct AVTreeNode *node = NULL;

    int ret;



    /* load glyph into s->face->glyph */

    if (FT_Load_Char(s->face, code, s->ft_load_flags))

        return AVERROR(EINVAL);



    /* save glyph */

    if (!(glyph = av_mallocz(sizeof(*glyph))) ||

        !(glyph->glyph = av_mallocz(sizeof(*glyph->glyph)))) {

        ret = AVERROR(ENOMEM);

        goto error;

    }

    glyph->code  = code;



    if (FT_Get_Glyph(s->face->glyph, glyph->glyph)) {

        ret = AVERROR(EINVAL);

        goto error;

    }

    if (s->borderw) {

        FT_Glyph border_glyph = *glyph->glyph;

        if (FT_Glyph_StrokeBorder(&border_glyph, s->stroker, 0, 0) ||

            FT_Glyph_To_Bitmap(&border_glyph, FT_RENDER_MODE_NORMAL, 0, 1)) {

            ret = AVERROR_EXTERNAL;

            goto error;

        }

        bitmapglyph = (FT_BitmapGlyph) border_glyph;

        glyph->border_bitmap = bitmapglyph->bitmap;

    }

    if (FT_Glyph_To_Bitmap(glyph->glyph, FT_RENDER_MODE_NORMAL, 0, 1)) {

        ret = AVERROR_EXTERNAL;

        goto error;

    }

    bitmapglyph = (FT_BitmapGlyph) *glyph->glyph;



    glyph->bitmap      = bitmapglyph->bitmap;

    glyph->bitmap_left = bitmapglyph->left;

    glyph->bitmap_top  = bitmapglyph->top;

    glyph->advance     = s->face->glyph->advance.x >> 6;



    /* measure text height to calculate text_height (or the maximum text height) */

    FT_Glyph_Get_CBox(*glyph->glyph, ft_glyph_bbox_pixels, &glyph->bbox);



    /* cache the newly created glyph */

    if (!(node = av_tree_node_alloc())) {

        ret = AVERROR(ENOMEM);

        goto error;

    }

    av_tree_insert(&s->glyphs, glyph, glyph_cmp, &node);



    if (glyph_ptr)

        *glyph_ptr = glyph;

    return 0;



error:

    if (glyph)

        av_freep(&glyph->glyph);

    av_freep(&glyph);

    av_freep(&node);

    return ret;

}
