static OSStatus ffat_decode_callback(AudioConverterRef converter, UInt32 *nb_packets,

                                     AudioBufferList *data,

                                     AudioStreamPacketDescription **packets,

                                     void *inctx)

{

    AVCodecContext *avctx = inctx;

    ATDecodeContext *at = avctx->priv_data;



    if (at->eof) {

        *nb_packets = 0;

        if (packets) {

            *packets = &at->pkt_desc;

            at->pkt_desc.mDataByteSize = 0;

        }

        return 0;

    }




    av_packet_move_ref(&at->in_pkt, &at->new_in_pkt);

    at->new_in_pkt.data = 0;

    at->new_in_pkt.size = 0;



    if (!at->in_pkt.data) {

        *nb_packets = 0;

        return 1;

    }



    data->mNumberBuffers              = 1;

    data->mBuffers[0].mNumberChannels = 0;

    data->mBuffers[0].mDataByteSize   = at->in_pkt.size;

    data->mBuffers[0].mData           = at->in_pkt.data;

    *nb_packets = 1;



    if (packets) {

        *packets = &at->pkt_desc;

        at->pkt_desc.mDataByteSize = at->in_pkt.size;

    }



    return 0;

}