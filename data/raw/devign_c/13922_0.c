int ff_alloc_packet2(AVCodecContext *avctx, AVPacket *avpkt, int size)

{

    if (size < 0 || avpkt->size < 0 || size > INT_MAX - FF_INPUT_BUFFER_PADDING_SIZE) {

        av_log(avctx, AV_LOG_ERROR, "Size %d invalid\n", size);

        return AVERROR(EINVAL);

    }



    if (avctx) {

        av_assert0(!avpkt->data || avpkt->data != avctx->internal->byte_buffer);

        if (!avpkt->data || avpkt->size < size) {

            av_fast_padded_malloc(&avctx->internal->byte_buffer, &avctx->internal->byte_buffer_size, size);

            avpkt->data = avctx->internal->byte_buffer;

            avpkt->size = avctx->internal->byte_buffer_size;

            avpkt->destruct = NULL;

        }

    }



    if (avpkt->data) {

        AVBufferRef *buf = avpkt->buf;

#if FF_API_DESTRUCT_PACKET

        void *destruct = avpkt->destruct;

#endif



        if (avpkt->size < size) {

            av_log(avctx, AV_LOG_ERROR, "User packet is too small (%d < %d)\n", avpkt->size, size);

            return AVERROR(EINVAL);

        }



        av_init_packet(avpkt);

#if FF_API_DESTRUCT_PACKET

        avpkt->destruct = destruct;

#endif

        avpkt->buf      = buf;

        avpkt->size     = size;

        return 0;

    } else {

        int ret = av_new_packet(avpkt, size);

        if (ret < 0)

            av_log(avctx, AV_LOG_ERROR, "Failed to allocate packet of size %d\n", size);

        return ret;

    }

}
