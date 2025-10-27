static int display_end_segment(AVCodecContext *avctx, void *data,

                               const uint8_t *buf, int buf_size)

{

    AVSubtitle    *sub = data;

    PGSSubContext *ctx = avctx->priv_data;



    /*

     *      The end display time is a timeout value and is only reached

     *      if the next subtitle is later then timeout or subtitle has

     *      not been cleared by a subsequent empty display command.

     */



    memset(sub, 0, sizeof(*sub));

    // Blank if last object_number was 0.

    // Note that this may be wrong for more complex subtitles.

    if (!ctx->presentation.object_number)

        return 1;

    sub->start_display_time = 0;

    sub->end_display_time   = 20000;

    sub->format             = 0;



    sub->rects     = av_mallocz(sizeof(*sub->rects));

    sub->rects[0]  = av_mallocz(sizeof(*sub->rects[0]));

    sub->num_rects = 1;



    sub->rects[0]->x    = ctx->presentation.x;

    sub->rects[0]->y    = ctx->presentation.y;

    sub->rects[0]->w    = ctx->picture.w;

    sub->rects[0]->h    = ctx->picture.h;

    sub->rects[0]->type = SUBTITLE_BITMAP;



    /* Process bitmap */

    sub->rects[0]->pict.linesize[0] = ctx->picture.w;



    if (ctx->picture.rle) {

        if (ctx->picture.rle_remaining_len)

            av_log(avctx, AV_LOG_ERROR, "RLE data length %u is %u bytes shorter than expected\n",

                   ctx->picture.rle_data_len, ctx->picture.rle_remaining_len);

        if(decode_rle(avctx, sub, ctx->picture.rle, ctx->picture.rle_data_len) < 0)

            return 0;

    }

    /* Allocate memory for colors */

    sub->rects[0]->nb_colors    = 256;

    sub->rects[0]->pict.data[1] = av_mallocz(AVPALETTE_SIZE);



    memcpy(sub->rects[0]->pict.data[1], ctx->clut, sub->rects[0]->nb_colors * sizeof(uint32_t));



    return 1;

}
