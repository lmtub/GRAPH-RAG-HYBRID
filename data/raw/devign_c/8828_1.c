int ff_v4l2_m2m_codec_end(AVCodecContext *avctx)

{

    V4L2m2mContext* s = avctx->priv_data;

    int ret;



    ret = ff_v4l2_context_set_status(&s->output, VIDIOC_STREAMOFF);

    if (ret)

            av_log(avctx, AV_LOG_ERROR, "VIDIOC_STREAMOFF %s\n", s->output.name);



    ret = ff_v4l2_context_set_status(&s->capture, VIDIOC_STREAMOFF);

    if (ret)

        av_log(avctx, AV_LOG_ERROR, "VIDIOC_STREAMOFF %s\n", s->capture.name);



    ff_v4l2_context_release(&s->output);



    if (atomic_load(&s->refcount))

        av_log(avctx, AV_LOG_ERROR, "ff_v4l2m2m_codec_end leaving pending buffers\n");



    ff_v4l2_context_release(&s->capture);

    sem_destroy(&s->refsync);



    /* release the hardware */

    if (close(s->fd) < 0 )

        av_log(avctx, AV_LOG_ERROR, "failure closing %s (%s)\n", s->devname, av_err2str(AVERROR(errno)));



    s->fd = -1;



    return 0;

}
