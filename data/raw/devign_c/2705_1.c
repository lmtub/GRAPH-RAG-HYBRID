AVFilterBufferRef *avfilter_default_get_audio_buffer(AVFilterLink *link, int perms,

                                                     enum AVSampleFormat sample_fmt, int size,

                                                     int64_t channel_layout, int planar)

{

    AVFilterBuffer *samples = av_mallocz(sizeof(AVFilterBuffer));

    AVFilterBufferRef *ref = NULL;

    int i, sample_size, chans_nb, bufsize, per_channel_size, step_size = 0;

    char *buf;



    if (!samples || !(ref = av_mallocz(sizeof(AVFilterBufferRef))))

        goto fail;



    ref->buf                   = samples;

    ref->format                = sample_fmt;



    ref->audio = av_mallocz(sizeof(AVFilterBufferRefAudioProps));

    if (!ref->audio)

        goto fail;



    ref->audio->channel_layout = channel_layout;

    ref->audio->size           = size;

    ref->audio->planar         = planar;



    /* make sure the buffer gets read permission or it's useless for output */

    ref->perms = perms | AV_PERM_READ;



    samples->refcount   = 1;

    samples->free       = ff_avfilter_default_free_buffer;



    sample_size = av_get_bits_per_sample_fmt(sample_fmt) >>3;

    chans_nb = av_get_channel_layout_nb_channels(channel_layout);



    per_channel_size = size/chans_nb;

    ref->audio->nb_samples = per_channel_size/sample_size;



    /* Set the number of bytes to traverse to reach next sample of a particular channel:

     * For planar, this is simply the sample size.

     * For packed, this is the number of samples * sample_size.

     */

    for (i = 0; i < chans_nb; i++)

        samples->linesize[i] = planar > 0 ? per_channel_size : sample_size;

    memset(&samples->linesize[chans_nb], 0, (8-chans_nb) * sizeof(samples->linesize[0]));



    /* Calculate total buffer size, round to multiple of 16 to be SIMD friendly */

    bufsize = (size + 15)&~15;

    buf = av_malloc(bufsize);

    if (!buf)

        goto fail;



    /* For planar, set the start point of each channel's data within the buffer

     * For packed, set the start point of the entire buffer only

     */

    samples->data[0] = buf;

    if (buf && planar) {

        for (i = 1; i < chans_nb; i++) {

            step_size += per_channel_size;

            samples->data[i] = buf + step_size;

        }

    } else {

        for (i = 1; i < chans_nb; i++)

            samples->data[i] = buf;

    }



    memset(&samples->data[chans_nb], 0, (8-chans_nb) * sizeof(samples->data[0]));



    memcpy(ref->data,     samples->data,     sizeof(ref->data));

    memcpy(ref->linesize, samples->linesize, sizeof(ref->linesize));



    return ref;



fail:

    av_free(buf);

    if (ref && ref->audio)

        av_free(ref->audio);

    av_free(ref);

    av_free(samples);

    return NULL;

}
