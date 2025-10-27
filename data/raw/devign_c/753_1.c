static int load_apply_palette(FFFrameSync *fs)

{

    AVFilterContext *ctx = fs->parent;

    AVFilterLink *inlink = ctx->inputs[0];

    PaletteUseContext *s = ctx->priv;

    AVFrame *master, *second, *out = NULL;

    int ret;



    // writable for error diffusal dithering

    ret = ff_framesync_dualinput_get_writable(fs, &master, &second);

    if (ret < 0)

        return ret;

    if (!master || !second) {

        ret = AVERROR_BUG;

        goto error;

    }

    if (!s->palette_loaded) {

        load_palette(s, second);

    }

    ret = apply_palette(inlink, master, &out);

    if (ret < 0)

        goto error;

    return ff_filter_frame(ctx->outputs[0], out);



error:

    av_frame_free(&master);

    av_frame_free(&second);

    return ret;

}
