static int ffat_encode(AVCodecContext *avctx, AVPacket *avpkt,

                       const AVFrame *frame, int *got_packet_ptr)

{

    ATDecodeContext *at = avctx->priv_data;

    OSStatus ret;



    AudioBufferList out_buffers = {

        .mNumberBuffers = 1,

        .mBuffers = {

            {

                .mNumberChannels = avctx->channels,

                .mDataByteSize = at->pkt_size,

            }

        }

    };

    AudioStreamPacketDescription out_pkt_desc = {0};



    if ((ret = ff_alloc_packet2(avctx, avpkt, at->pkt_size, 0)) < 0)

        return ret;



    av_frame_unref(&at->new_in_frame);



    if (frame) {

        if ((ret = ff_af_queue_add(&at->afq, frame)) < 0)

            return ret;

        if ((ret = av_frame_ref(&at->new_in_frame, frame)) < 0)

            return ret;

    } else {

        at->eof = 1;

    }



    out_buffers.mBuffers[0].mData = avpkt->data;



    *got_packet_ptr = avctx->frame_size / at->frame_size;



    ret = AudioConverterFillComplexBuffer(at->converter, ffat_encode_callback, avctx,

                                          got_packet_ptr, &out_buffers,

                                          (avctx->frame_size > at->frame_size) ? NULL : &out_pkt_desc);

    if ((!ret || ret == 1) && *got_packet_ptr) {

        avpkt->size = out_buffers.mBuffers[0].mDataByteSize;

        ff_af_queue_remove(&at->afq, out_pkt_desc.mVariableFramesInPacket ?

                                     out_pkt_desc.mVariableFramesInPacket :

                                     avctx->frame_size,

                           &avpkt->pts,

                           &avpkt->duration);

    } else if (ret && ret != 1) {

        av_log(avctx, AV_LOG_WARNING, "Encode error: %i\n", ret);

    }



    return 0;

}
