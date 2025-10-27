int av_write_frame(AVFormatContext *s, AVPacket *pkt)

{

    int ret = compute_pkt_fields2(s, s->streams[pkt->stream_index], pkt);



    if(ret<0 && !(s->oformat->flags & AVFMT_NOTIMESTAMPS))

        return ret;



    ret= s->oformat->write_packet(s, pkt);

    if(!ret)

        ret= url_ferror(s->pb);

    return ret;

}
