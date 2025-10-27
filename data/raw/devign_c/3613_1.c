static int config_output(AVFilterLink *outlink)

{

    AVFilterContext *ctx = outlink->src;

    LIBVMAFContext *s = ctx->priv;

    AVFilterLink *mainlink = ctx->inputs[0];

    int ret;



    outlink->w = mainlink->w;

    outlink->h = mainlink->h;

    outlink->time_base = mainlink->time_base;

    outlink->sample_aspect_ratio = mainlink->sample_aspect_ratio;

    outlink->frame_rate = mainlink->frame_rate;

    if ((ret = ff_dualinput_init(ctx, &s->dinput)) < 0)

        return ret;



    return 0;

}
