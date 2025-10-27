int attribute_align_arg av_buffersink_get_frame_flags(AVFilterContext *ctx, AVFrame *frame, int flags)

{

    BufferSinkContext *buf = ctx->priv;

    AVFilterLink *inlink = ctx->inputs[0];

    int ret;

    AVFrame *cur_frame;



    /* no picref available, fetch it from the filterchain */

    if (!av_fifo_size(buf->fifo)) {

        if (flags & AV_BUFFERSINK_FLAG_NO_REQUEST)

            return AVERROR(EAGAIN);

        if ((ret = ff_request_frame(inlink)) < 0)

            return ret;

    }



    if (!av_fifo_size(buf->fifo))

        return AVERROR(EINVAL);



    if (flags & AV_BUFFERSINK_FLAG_PEEK) {

        cur_frame = *((AVFrame **)av_fifo_peek2(buf->fifo, 0));

        av_frame_ref(frame, cur_frame); /* TODO check failure */

    } else {

        av_fifo_generic_read(buf->fifo, &cur_frame, sizeof(cur_frame), NULL);

        av_frame_move_ref(frame, cur_frame);

        av_frame_free(&cur_frame);

    }



    return 0;

}
