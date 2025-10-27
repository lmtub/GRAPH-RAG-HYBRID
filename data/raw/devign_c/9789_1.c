static int channelmap_query_formats(AVFilterContext *ctx)

{

    ChannelMapContext *s = ctx->priv;



    ff_set_common_formats(ctx, ff_planar_sample_fmts());

    ff_set_common_samplerates(ctx, ff_all_samplerates());

    ff_channel_layouts_ref(ff_all_channel_layouts(), &ctx->inputs[0]->out_channel_layouts);

    ff_channel_layouts_ref(s->channel_layouts,       &ctx->outputs[0]->in_channel_layouts);



    return 0;

}
