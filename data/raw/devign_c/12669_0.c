static int query_formats(AVFilterContext *ctx)

{

    static const enum AVPixelFormat pix_fmts[] = {

        AV_PIX_FMT_GRAY8,

        AV_PIX_FMT_YUV410P, AV_PIX_FMT_YUV411P,

        AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUV422P,

        AV_PIX_FMT_YUV440P, AV_PIX_FMT_YUV444P,

        AV_PIX_FMT_YUVJ420P, AV_PIX_FMT_YUVJ422P,

        AV_PIX_FMT_YUVJ440P, AV_PIX_FMT_YUVJ444P,

        AV_PIX_FMT_YUVJ411P,

        AV_PIX_FMT_YUVA444P, AV_PIX_FMT_YUVA422P, AV_PIX_FMT_YUVA420P,

        AV_PIX_FMT_NONE

    };



    AVFilterFormats *fmts_list = ff_make_format_list(pix_fmts);

    if (!fmts_list)

        return AVERROR(ENOMEM);

    ff_set_common_formats(ctx, fmts_list);



    return 0;

}
