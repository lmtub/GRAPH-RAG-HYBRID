static int dvbsub_display_end_segment(AVCodecContext *avctx, const uint8_t *buf,

                                      int buf_size, AVSubtitle *sub)

{

    DVBSubContext *ctx = avctx->priv_data;

    DVBSubDisplayDefinition *display_def = ctx->display_definition;



    DVBSubRegion *region;

    DVBSubRegionDisplay *display;

    AVSubtitleRect *rect;

    DVBSubCLUT *clut;

    uint32_t *clut_table;

    int i;

    int offset_x=0, offset_y=0;



    sub->rects = NULL;

    sub->start_display_time = 0;

    sub->end_display_time = ctx->time_out * 1000;

    sub->format = 0;



    if (display_def) {

        offset_x = display_def->x;

        offset_y = display_def->y;

    }



    sub->num_rects = ctx->display_list_size;

    if (sub->num_rects <= 0)

        return AVERROR_INVALIDDATA;



    sub->rects = av_mallocz_array(sub->num_rects * sub->num_rects,

                                  sizeof(*sub->rects));

    if (!sub->rects)

        return AVERROR(ENOMEM);



    i = 0;



    for (display = ctx->display_list; display; display = display->next) {

        region = get_region(ctx, display->region_id);

        rect = sub->rects[i];



        if (!region)

            continue;



        rect->x = display->x_pos + offset_x;

        rect->y = display->y_pos + offset_y;

        rect->w = region->width;

        rect->h = region->height;

        rect->nb_colors = 16;

        rect->type      = SUBTITLE_BITMAP;

        rect->linesize[0] = region->width;



        clut = get_clut(ctx, region->clut);



        if (!clut)

            clut = &default_clut;



        switch (region->depth) {

        case 2:

            clut_table = clut->clut4;

            break;

        case 8:

            clut_table = clut->clut256;

            break;

        case 4:

        default:

            clut_table = clut->clut16;

            break;

        }



        rect->data[1] = av_mallocz(AVPALETTE_SIZE);

        if (!rect->data[1]) {

            av_free(sub->rects);

            return AVERROR(ENOMEM);

        }

        memcpy(rect->data[1], clut_table, (1 << region->depth) * sizeof(uint32_t));



        rect->data[0] = av_malloc(region->buf_size);

        if (!rect->data[0]) {

            av_free(rect->data[1]);

            av_free(sub->rects);

            return AVERROR(ENOMEM);

        }

        memcpy(rect->data[0], region->pbuf, region->buf_size);



#if FF_API_AVPICTURE

FF_DISABLE_DEPRECATION_WARNINGS

{

        int j;

        for (j = 0; j < 4; j++) {

            rect->pict.data[j] = rect->data[j];

            rect->pict.linesize[j] = rect->linesize[j];

        }

}

FF_ENABLE_DEPRECATION_WARNINGS

#endif



        i++;

    }



    sub->num_rects = i;



#ifdef DEBUG

    save_display_set(ctx);

#endif



    return 1;

}
