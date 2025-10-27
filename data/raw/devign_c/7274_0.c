AVFormatContext *avformat_alloc_context(void)

{

    AVFormatContext *ic;

    ic = av_malloc(sizeof(AVFormatContext));

    if (!ic) return ic;

    avformat_get_context_defaults(ic);

    ic->av_class = &av_format_context_class;

    return ic;

}
