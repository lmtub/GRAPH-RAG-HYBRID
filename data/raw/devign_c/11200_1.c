static av_cold int decimate_init(AVFilterContext *ctx)

{

    DecimateContext *dm = ctx->priv;

    AVFilterPad pad = {

        .name         = av_strdup("main"),

        .type         = AVMEDIA_TYPE_VIDEO,

        .filter_frame = filter_frame,

        .config_props = config_input,

    };



    if (!pad.name)

        return AVERROR(ENOMEM);

    ff_insert_inpad(ctx, INPUT_MAIN, &pad);



    if (dm->ppsrc) {

        pad.name = av_strdup("clean_src");

        pad.config_props = NULL;

        if (!pad.name)

            return AVERROR(ENOMEM);

        ff_insert_inpad(ctx, INPUT_CLEANSRC, &pad);

    }



    if ((dm->blockx & (dm->blockx - 1)) ||

        (dm->blocky & (dm->blocky - 1))) {

        av_log(ctx, AV_LOG_ERROR, "blockx and blocky settings must be power of two\n");

        return AVERROR(EINVAL);

    }



    dm->start_pts = AV_NOPTS_VALUE;



    return 0;

}
