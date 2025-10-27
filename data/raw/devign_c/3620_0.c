static int oma_read_packet(AVFormatContext *s, AVPacket *pkt)

{

    OMAContext *oc = s->priv_data;

    int packet_size = s->streams[0]->codec->block_align;

    int ret = av_get_packet(s->pb, pkt, packet_size);



    if (ret <= 0)

        return AVERROR(EIO);



    pkt->stream_index = 0;



    if (oc->encrypted) {

        /* previous unencrypted block saved in IV for

         * the next packet (CBC mode) */

        av_des_crypt(&oc->av_des, pkt->data, pkt->data,

                     (packet_size >> 3), oc->iv, 1);

    }



    return ret;

}
