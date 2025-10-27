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



    sub->end_display_time = ctx->time_out * 1000;



    if (display_def) {

        offset_x = display_def->x;

        offset_y = display_def->y;

    }



    sub->num_rects = ctx->display_list_size;



    if (sub->num_rects > 0){

        sub->rects = av_mallocz(sizeof(*sub->rects) * sub->num_rects);

        for(i=0; i<sub->num_rects; i++)

            sub->rects[i] = av_mallocz(sizeof(*sub->rects[i]));



    i = 0;



    for (display = ctx->display_list; display; display = display->next) {

        region = get_region(ctx, display->region_id);



        if (!region)

            continue;



        if (!region->dirty)

            continue;



        rect = sub->rects[i];

        rect->x = display->x_pos + offset_x;

        rect->y = display->y_pos + offset_y;

        rect->w = region->width;

        rect->h = region->height;

        rect->nb_colors = (1 << region->depth);

        rect->type      = SUBTITLE_BITMAP;

        rect->pict.linesize[0] = region->width;



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



        rect->pict.data[1] = av_mallocz(AVPALETTE_SIZE);

        memcpy(rect->pict.data[1], clut_table, (1 << region->depth) * sizeof(uint32_t));



        rect->pict.data[0] = av_malloc(region->buf_size);

        memcpy(rect->pict.data[0], region->pbuf, region->buf_size);



        i++;

    }



    sub->num_rects = i;

    }

#ifdef DEBUG

    save_display_set(ctx);

#endif



    return 1;

}
