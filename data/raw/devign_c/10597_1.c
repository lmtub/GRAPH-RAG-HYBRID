static av_cold int dilate_init(AVFilterContext *ctx, const char *args)

{

    OCVContext *ocv = ctx->priv;

    DilateContext *dilate = ocv->priv;

    char default_kernel_str[] = "3x3+0x0/rect";

    char *kernel_str;

    const char *buf = args;

    int ret;



    dilate->nb_iterations = 1;



    if (args)

        kernel_str = av_get_token(&buf, "|");

    if ((ret = parse_iplconvkernel(&dilate->kernel,

                                   *kernel_str ? kernel_str : default_kernel_str,

                                   ctx)) < 0)

        return ret;

    av_free(kernel_str);



    sscanf(buf, "|%d", &dilate->nb_iterations);

    av_log(ctx, AV_LOG_VERBOSE, "iterations_nb:%d\n", dilate->nb_iterations);

    if (dilate->nb_iterations <= 0) {

        av_log(ctx, AV_LOG_ERROR, "Invalid non-positive value '%d' for nb_iterations\n",

               dilate->nb_iterations);

        return AVERROR(EINVAL);

    }

    return 0;

}
