int ff_qsv_encode(AVCodecContext *avctx, QSVEncContext *q,

                  AVPacket *pkt, const AVFrame *frame, int *got_packet)

{

    int ret;



    ret = encode_frame(avctx, q, frame);

    if (ret < 0)

        return ret;



    if (!av_fifo_space(q->async_fifo) ||

        (!frame && av_fifo_size(q->async_fifo))) {

        AVPacket new_pkt;

        mfxBitstream *bs;

        mfxSyncPoint sync;



        av_fifo_generic_read(q->async_fifo, &new_pkt, sizeof(new_pkt), NULL);

        av_fifo_generic_read(q->async_fifo, &sync,    sizeof(sync),    NULL);

        av_fifo_generic_read(q->async_fifo, &bs,      sizeof(bs),      NULL);



        do {

            ret = MFXVideoCORE_SyncOperation(q->session, sync, 1000);

        } while (ret == MFX_WRN_IN_EXECUTION);



        new_pkt.dts  = av_rescale_q(bs->DecodeTimeStamp, (AVRational){1, 90000}, avctx->time_base);

        new_pkt.pts  = av_rescale_q(bs->TimeStamp,       (AVRational){1, 90000}, avctx->time_base);

        new_pkt.size = bs->DataLength;



        if (bs->FrameType & MFX_FRAMETYPE_IDR ||

            bs->FrameType & MFX_FRAMETYPE_xIDR)

            new_pkt.flags |= AV_PKT_FLAG_KEY;



#if FF_API_CODED_FRAME

FF_DISABLE_DEPRECATION_WARNINGS

        if (bs->FrameType & MFX_FRAMETYPE_I || bs->FrameType & MFX_FRAMETYPE_xI)

            avctx->coded_frame->pict_type = AV_PICTURE_TYPE_I;

        else if (bs->FrameType & MFX_FRAMETYPE_P || bs->FrameType & MFX_FRAMETYPE_xP)

            avctx->coded_frame->pict_type = AV_PICTURE_TYPE_P;

        else if (bs->FrameType & MFX_FRAMETYPE_B || bs->FrameType & MFX_FRAMETYPE_xB)

            avctx->coded_frame->pict_type = AV_PICTURE_TYPE_B;

FF_ENABLE_DEPRECATION_WARNINGS

#endif



        av_freep(&bs);



        if (pkt->data) {

            if (pkt->size < new_pkt.size) {

                av_log(avctx, AV_LOG_ERROR, "Submitted buffer not large enough: %d < %d\n",

                       pkt->size, new_pkt.size);

                av_packet_unref(&new_pkt);

                return AVERROR(EINVAL);

            }



            memcpy(pkt->data, new_pkt.data, new_pkt.size);

            pkt->size = new_pkt.size;



            ret = av_packet_copy_props(pkt, &new_pkt);

            av_packet_unref(&new_pkt);

            if (ret < 0)

                return ret;

        } else

            *pkt = new_pkt;



        *got_packet = 1;

    }



    return 0;

}
