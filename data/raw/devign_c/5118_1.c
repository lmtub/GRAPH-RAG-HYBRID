static int update_thread_context(AVCodecContext *dst, const AVCodecContext *src)

{

    PNGDecContext *psrc = src->priv_data;

    PNGDecContext *pdst = dst->priv_data;

    int ret;



    if (dst == src)

        return 0;



    ff_thread_release_buffer(dst, &pdst->picture);

    if (psrc->picture.f->data[0] &&

        (ret = ff_thread_ref_frame(&pdst->picture, &psrc->picture)) < 0)

        return ret;

    if (CONFIG_APNG_DECODER && dst->codec_id == AV_CODEC_ID_APNG) {

        pdst->width             = psrc->width;

        pdst->height            = psrc->height;

        pdst->bit_depth         = psrc->bit_depth;

        pdst->color_type        = psrc->color_type;

        pdst->compression_type  = psrc->compression_type;

        pdst->interlace_type    = psrc->interlace_type;

        pdst->filter_type       = psrc->filter_type;

        pdst->cur_w = psrc->cur_w;

        pdst->cur_h = psrc->cur_h;

        pdst->x_offset = psrc->x_offset;

        pdst->y_offset = psrc->y_offset;

        pdst->has_trns = psrc->has_trns;

        memcpy(pdst->transparent_color_be, psrc->transparent_color_be, sizeof(pdst->transparent_color_be));



        pdst->dispose_op = psrc->dispose_op;



        memcpy(pdst->palette, psrc->palette, sizeof(pdst->palette));



        pdst->state |= psrc->state & (PNG_IHDR | PNG_PLTE);



        ff_thread_release_buffer(dst, &pdst->last_picture);

        if (psrc->last_picture.f->data[0] &&

            (ret = ff_thread_ref_frame(&pdst->last_picture, &psrc->last_picture)) < 0)

            return ret;



        ff_thread_release_buffer(dst, &pdst->previous_picture);

        if (psrc->previous_picture.f->data[0] &&

            (ret = ff_thread_ref_frame(&pdst->previous_picture, &psrc->previous_picture)) < 0)

            return ret;

    }



    return 0;

}
