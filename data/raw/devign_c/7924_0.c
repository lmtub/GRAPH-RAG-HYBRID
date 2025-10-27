AVFilterBufferRef *avfilter_get_audio_buffer(AVFilterLink *link, int perms,

                                             enum AVSampleFormat sample_fmt, int size,

                                             int64_t channel_layout, int planar)

{

    AVFilterBufferRef *ret = NULL;



    if (link->dstpad->get_audio_buffer)

        ret = link->dstpad->get_audio_buffer(link, perms, sample_fmt, size, channel_layout, planar);



    if (!ret)

        ret = avfilter_default_get_audio_buffer(link, perms, sample_fmt, size, channel_layout, planar);



    if (ret)

        ret->type = AVMEDIA_TYPE_AUDIO;



    return ret;

}
