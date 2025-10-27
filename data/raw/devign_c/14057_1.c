static int v4l2_receive_frame(AVCodecContext *avctx, AVFrame *frame)

{

    V4L2m2mContext *s = avctx->priv_data;

    V4L2Context *const capture = &s->capture;

    V4L2Context *const output = &s->output;

    AVPacket avpkt = {0};

    int ret;



    ret = ff_decode_get_packet(avctx, &avpkt);

    if (ret < 0 && ret != AVERROR_EOF)

        return ret;



    if (s->draining)

        goto dequeue;



    ret = ff_v4l2_context_enqueue_packet(output, &avpkt);

    if (ret < 0) {

        if (ret != AVERROR(ENOMEM))

           return ret;

        /* no input buffers available, continue dequeing */

    }



    if (avpkt.size) {

        ret = v4l2_try_start(avctx);

        if (ret)

            return 0;

    }



dequeue:

    return ff_v4l2_context_dequeue_frame(capture, frame);

}
