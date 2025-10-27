static int vqf_read_packet(AVFormatContext *s, AVPacket *pkt)

{

    VqfContext *c = s->priv_data;

    int ret;

    int size = (c->frame_bit_len - c->remaining_bits + 7)>>3;



    if (av_new_packet(pkt, size+2) < 0)

        return AVERROR(EIO);



    pkt->pos          = avio_tell(s->pb);

    pkt->stream_index = 0;

    pkt->duration     = 1;



    pkt->data[0] = 8 - c->remaining_bits; // Number of bits to skip

    pkt->data[1] = c->last_frame_bits;

    ret = avio_read(s->pb, pkt->data+2, size);



    if (ret<=0) {

        av_free_packet(pkt);

        return AVERROR(EIO);

    }



    c->last_frame_bits = pkt->data[size+1];

    c->remaining_bits  = (size << 3) - c->frame_bit_len + c->remaining_bits;



    return size+2;

}
