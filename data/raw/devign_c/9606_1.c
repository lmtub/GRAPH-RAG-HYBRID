theora_gptopts(AVFormatContext *ctx, int idx, uint64_t gp, int64_t *dts)

{

    struct ogg *ogg = ctx->priv_data;

    struct ogg_stream *os = ogg->streams + idx;

    struct theora_params *thp = os->private;

    uint64_t iframe = gp >> thp->gpshift;

    uint64_t pframe = gp & thp->gpmask;



    if (thp->version < 0x030201)

        iframe++;



    if(!pframe)

        os->pflags |= AV_PKT_FLAG_KEY;



    if (dts)

        *dts = iframe + pframe;



    return iframe + pframe;

}
