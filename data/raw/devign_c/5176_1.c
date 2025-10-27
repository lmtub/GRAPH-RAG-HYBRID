static int hwmap_filter_frame(AVFilterLink *link, AVFrame *input)

{

    AVFilterContext *avctx = link->dst;

    AVFilterLink  *outlink = avctx->outputs[0];

    HWMapContext      *ctx = avctx->priv;

    AVFrame *map = NULL;

    int err;



    av_log(ctx, AV_LOG_DEBUG, "Filter input: %s, %ux%u (%"PRId64").\n",

           av_get_pix_fmt_name(input->format),

           input->width, input->height, input->pts);



    map = av_frame_alloc();

    if (!map) {

        err = AVERROR(ENOMEM);

        goto fail;

    }



    map->format = outlink->format;

    map->hw_frames_ctx = av_buffer_ref(ctx->hwframes_ref);

    if (!map->hw_frames_ctx) {

        err = AVERROR(ENOMEM);

        goto fail;

    }



    if (ctx->map_backwards && !input->hw_frames_ctx) {

        // If we mapped backwards from hardware to software, we need

        // to attach the hardware frame context to the input frame to

        // make the mapping visible to av_hwframe_map().

        input->hw_frames_ctx = av_buffer_ref(ctx->hwframes_ref);

        if (!input->hw_frames_ctx) {

            err = AVERROR(ENOMEM);

            goto fail;

        }

    }



    err = av_hwframe_map(map, input, ctx->mode);

    if (err < 0) {

        av_log(avctx, AV_LOG_ERROR, "Failed to map frame: %d.\n", err);

        goto fail;

    }



    err = av_frame_copy_props(map, input);

    if (err < 0)

        goto fail;



    av_frame_free(&input);



    av_log(ctx, AV_LOG_DEBUG, "Filter output: %s, %ux%u (%"PRId64").\n",

           av_get_pix_fmt_name(map->format),

           map->width, map->height, map->pts);



    return ff_filter_frame(outlink, map);



fail:

    av_frame_free(&input);

    av_frame_free(&map);

    return err;

}
