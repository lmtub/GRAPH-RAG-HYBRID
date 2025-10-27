int av_write_frame(AVFormatContext *s, AVPacket *pkt)

{

    int ret;



    compute_pkt_fields2(s->streams[pkt->stream_index], pkt);

    

    truncate_ts(s->streams[pkt->stream_index], pkt);



    ret= s->oformat->write_packet(s, pkt);

    if(!ret)

        ret= url_ferror(&s->pb);

    return ret;

}
