static int rso_read_packet(AVFormatContext *s, AVPacket *pkt)

{

    int bps = av_get_bits_per_sample(s->streams[0]->codec->codec_id);

    int ret = av_get_packet(s->pb, pkt, BLOCK_SIZE * bps >> 3);



    if (ret < 0)

        return ret;




    pkt->stream_index = 0;



    /* note: we need to modify the packet size here to handle the last packet */

    pkt->size = ret;



    return 0;

}