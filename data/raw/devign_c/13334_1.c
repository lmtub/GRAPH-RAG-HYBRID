AVFilterBufferRef *avfilter_default_get_audio_buffer(AVFilterLink *link, int perms,

                                                     int nb_samples)

{

    AVFilterBufferRef *samplesref = NULL;

    int linesize[8];

    uint8_t *data[8];

    int nb_channels = av_get_channel_layout_nb_channels(link->channel_layout);



    /* Calculate total buffer size, round to multiple of 16 to be SIMD friendly */

    if (av_samples_alloc(data, linesize,

                         nb_channels, nb_samples, link->format,

                         16) < 0)

        return NULL;



    samplesref =

        avfilter_get_audio_buffer_ref_from_arrays(data, linesize, perms,

                                                  nb_samples, link->format,

                                                  link->channel_layout, link->planar);

    if (!samplesref) {

        av_free(data[0]);

        return NULL;

    }



    return samplesref;

}
