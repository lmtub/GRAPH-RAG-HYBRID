static av_cold int init(AVFilterContext *ctx, const char *args, void *opaque)

{

    UnsharpContext *unsharp = ctx->priv;

    int lmsize_x = 5, cmsize_x = 0;

    int lmsize_y = 5, cmsize_y = 0;

    double lamount = 1.0f, camount = 0.0f;



    if (args)

        sscanf(args, "%d:%d:%lf:%d:%d:%lf", &lmsize_x, &lmsize_y, &lamount,

                                            &cmsize_x, &cmsize_y, &camount);



    if (lmsize_x < 2 || lmsize_y < 2 || cmsize_x < 2 || cmsize_y < 2) {

        av_log(ctx, AV_LOG_ERROR,

               "Invalid value <2 for lmsize_x:%d or lmsize_y:%d or cmsize_x:%d or cmsize_y:%d\n",

               lmsize_x, lmsize_y, cmsize_x, cmsize_y);

        return AVERROR(EINVAL);

    }



    set_filter_param(&unsharp->luma,   lmsize_x, lmsize_y, lamount);

    set_filter_param(&unsharp->chroma, cmsize_x, cmsize_y, camount);



    return 0;

}
