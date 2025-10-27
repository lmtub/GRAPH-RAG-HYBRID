AVFilterBufferRef *avfilter_get_audio_buffer_ref_from_frame(const AVFrame *frame,

                                                            int perms)

{

    AVFilterBufferRef *samplesref =

        avfilter_get_audio_buffer_ref_from_arrays((uint8_t **)frame->data, frame->linesize[0], perms,

                                                  frame->nb_samples, frame->format,

                                                  av_frame_get_channel_layout(frame));

    if (!samplesref)

        return NULL;

    if (avfilter_copy_frame_props(samplesref, frame) < 0) {

        samplesref->buf->data[0] = NULL;

        avfilter_unref_bufferp(&samplesref);

    }

    return samplesref;

}
