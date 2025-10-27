int av_buffersrc_add_frame(AVFilterContext *ctx, AVFrame *frame)

{

    BufferSourceContext *s = ctx->priv;

    AVFrame *copy;

    int ret;



    if (!frame) {

        s->eof = 1;

        return 0;

    } else if (s->eof)

        return AVERROR(EINVAL);



    switch (ctx->outputs[0]->type) {

    case AVMEDIA_TYPE_VIDEO:

        CHECK_VIDEO_PARAM_CHANGE(ctx, s, frame->width, frame->height,

                                 frame->format);

        break;

    case AVMEDIA_TYPE_AUDIO:

        CHECK_AUDIO_PARAM_CHANGE(ctx, s, frame->sample_rate, frame->channel_layout,

                                 frame->format);

        break;

    default:

        return AVERROR(EINVAL);

    }



    if (!av_fifo_space(s->fifo) &&

        (ret = av_fifo_realloc2(s->fifo, av_fifo_size(s->fifo) +

                                         sizeof(copy))) < 0)

        return ret;



    if (!(copy = av_frame_alloc()))

        return AVERROR(ENOMEM);

    av_frame_move_ref(copy, frame);



    if ((ret = av_fifo_generic_write(s->fifo, &copy, sizeof(copy), NULL)) < 0) {

        av_frame_move_ref(frame, copy);

        av_frame_free(&copy);

        return ret;

    }



    return 0;

}
