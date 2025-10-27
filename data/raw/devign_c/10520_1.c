static int prepare_packet(AVPacket *pkt,const FailingMuxerPacketData *pkt_data, int64_t pts)

{

    int ret;

    FailingMuxerPacketData *data = av_malloc(sizeof(*data));




    memcpy(data, pkt_data, sizeof(FailingMuxerPacketData));

    ret = av_packet_from_data(pkt, (uint8_t*) data, sizeof(*data));



    pkt->pts = pkt->dts = pts;

    pkt->duration = 1;



    return ret;
