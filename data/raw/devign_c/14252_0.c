int av_write_trailer(AVFormatContext *s)

{

    int ret;

    

    while(s->packet_buffer){

        int ret;

        AVPacketList *pktl= s->packet_buffer;



//av_log(s, AV_LOG_DEBUG, "write_trailer st:%d dts:%lld\n", pktl->pkt.stream_index, pktl->pkt.dts);

        truncate_ts(s->streams[pktl->pkt.stream_index], &pktl->pkt);

        ret= s->oformat->write_packet(s, &pktl->pkt);

        

        s->packet_buffer= pktl->next;        



        av_free_packet(&pktl->pkt);

        av_freep(&pktl);

        

        if(ret<0)

            return ret;

    }



    ret = s->oformat->write_trailer(s);

    av_freep(&s->priv_data);

    return ret;

}
