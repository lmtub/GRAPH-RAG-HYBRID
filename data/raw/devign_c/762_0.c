int av_vsrc_buffer_add_frame(AVFilterContext *buffer_filter, AVFrame *frame,

                             int64_t pts, AVRational pixel_aspect)

{

    BufferSourceContext *c = buffer_filter->priv;

    AVFilterBufferRef *buf;

    int ret;



    if (!buf) {

        c->eof = 1;

        return 0;

    } else if (c->eof)

        return AVERROR(EINVAL);



    if (!av_fifo_space(c->fifo) &&

        (ret = av_fifo_realloc2(c->fifo, av_fifo_size(c->fifo) +

                                         sizeof(buf))) < 0)

        return ret;



    CHECK_PARAM_CHANGE(buffer_filter, c, frame->width, frame->height, frame->format);



    buf = avfilter_get_video_buffer(buffer_filter->outputs[0], AV_PERM_WRITE,

                                    c->w, c->h);

    av_image_copy(buf->data, buf->linesize, frame->data, frame->linesize,

                  c->pix_fmt, c->w, c->h);



    avfilter_copy_frame_props(buf, frame);

    buf->pts                    = pts;

    buf->video->pixel_aspect    = pixel_aspect;



    if ((ret = av_fifo_generic_write(c->fifo, &buf, sizeof(buf), NULL)) < 0) {

        avfilter_unref_buffer(buf);

        return ret;

    }



    return 0;

}
