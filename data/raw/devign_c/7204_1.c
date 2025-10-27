static int copy_packet_data(AVPacket *pkt, AVPacket *src, int dup)

{

    pkt->data      = NULL;

    pkt->side_data = NULL;

    if (pkt->buf) {

        AVBufferRef *ref = av_buffer_ref(src->buf);

        if (!ref)

            return AVERROR(ENOMEM);

        pkt->buf  = ref;

        pkt->data = ref->data;

    } else {

        DUP_DATA(pkt->data, src->data, pkt->size, 1, ALLOC_BUF);

    }

#if FF_API_DESTRUCT_PACKET

FF_DISABLE_DEPRECATION_WARNINGS

    pkt->destruct = dummy_destruct_packet;

FF_ENABLE_DEPRECATION_WARNINGS

#endif

    if (pkt->side_data_elems && dup)

        pkt->side_data = src->side_data;

    if (pkt->side_data_elems && !dup) {

        return av_copy_packet_side_data(pkt, src);

    }

    return 0;



failed_alloc:

    av_destruct_packet(pkt);

    return AVERROR(ENOMEM);

}
