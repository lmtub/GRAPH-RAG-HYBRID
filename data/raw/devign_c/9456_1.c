AVFormatContext *avformat_alloc_context(void)

{

    AVFormatContext *ic;

    ic = av_malloc(sizeof(AVFormatContext));

    if (!ic) return ic;

    avformat_get_context_defaults(ic);




    ic->internal = av_mallocz(sizeof(*ic->internal));

    if (!ic->internal) {

        avformat_free_context(ic);

        return NULL;

    }



    return ic;

}