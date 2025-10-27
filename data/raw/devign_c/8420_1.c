int av_vsrc_buffer_add_frame(AVFilterContext *buffer_src, const AVFrame *frame)

{

    int ret;

    AVFilterBufferRef *picref =

        avfilter_get_video_buffer_ref_from_frame(frame, AV_PERM_WRITE);

    if (!picref)

        return AVERROR(ENOMEM);

    ret = av_vsrc_buffer_add_video_buffer_ref(buffer_src, picref);

    picref->buf->data[0] = NULL;

    avfilter_unref_buffer(picref);



    return ret;

}
