static int Faac_encode_frame(AVCodecContext *avctx, AVPacket *avpkt,

                             const AVFrame *frame, int *got_packet_ptr)

{

    FaacAudioContext *s = avctx->priv_data;

    int bytes_written, ret;

    int num_samples  = frame ? frame->nb_samples : 0;

    void *samples    = frame ? frame->data[0]    : NULL;



    if ((ret = ff_alloc_packet2(avctx, avpkt, (7 + 768) * avctx->channels))) {

        av_log(avctx, AV_LOG_ERROR, "Error getting output packet\n");

        return ret;

    }



    bytes_written = faacEncEncode(s->faac_handle, samples,

                                  num_samples * avctx->channels,

                                  avpkt->data, avpkt->size);

    if (bytes_written < 0) {

        av_log(avctx, AV_LOG_ERROR, "faacEncEncode() error\n");

        return bytes_written;

    }



    /* add current frame to the queue */

    if (frame) {

        if ((ret = ff_af_queue_add(&s->afq, frame)) < 0)

            return ret;

    }



    if (!bytes_written)

        return 0;



    /* Get the next frame pts/duration */

    ff_af_queue_remove(&s->afq, avctx->frame_size, &avpkt->pts,

                       &avpkt->duration);



    avpkt->size = bytes_written;

    *got_packet_ptr = 1;

    return 0;

}
