static int compat_read(AVFilterContext *ctx, AVFilterBufferRef **pbuf, int nb_samples, int flags)

{

    AVFilterBufferRef *buf;

    AVFrame *frame;

    int ret;



    if (!pbuf)

        return ff_poll_frame(ctx->inputs[0]);



    frame = av_frame_alloc();

    if (!frame)

        return AVERROR(ENOMEM);



    if (!nb_samples)

        ret = av_buffersink_get_frame_flags(ctx, frame, flags);

    else

        ret = av_buffersink_get_samples(ctx, frame, nb_samples);



    if (ret < 0)

        goto fail;



    if (ctx->inputs[0]->type == AVMEDIA_TYPE_VIDEO) {

        buf = avfilter_get_video_buffer_ref_from_arrays(frame->data, frame->linesize,

                                                        AV_PERM_READ,

                                                        frame->width, frame->height,

                                                        frame->format);

    } else {

        buf = avfilter_get_audio_buffer_ref_from_arrays(frame->extended_data,

                                                        frame->linesize[0], AV_PERM_READ,

                                                        frame->nb_samples,

                                                        frame->format,

                                                        frame->channel_layout);

    }

    if (!buf) {

        ret = AVERROR(ENOMEM);

        goto fail;

    }



    avfilter_copy_frame_props(buf, frame);



    buf->buf->priv = frame;

    buf->buf->free = compat_free_buffer;



    *pbuf = buf;



    return 0;

fail:

    av_frame_free(&frame);

    return ret;

}
