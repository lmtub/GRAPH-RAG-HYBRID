static int mp3_read_packet(AVFormatContext *s, AVPacket *pkt)

{

    int ret;



    ret = av_get_packet(s->pb, pkt, MP3_PACKET_SIZE);



    pkt->stream_index = 0;

    if (ret <= 0) {

        return AVERROR(EIO);

    }



    if (ret > ID3v1_TAG_SIZE &&

        memcmp(&pkt->data[ret - ID3v1_TAG_SIZE], "TAG", 3) == 0)

        ret -= ID3v1_TAG_SIZE;



    /* note: we need to modify the packet size here to handle the last

       packet */

    pkt->size = ret;

    return ret;

}
