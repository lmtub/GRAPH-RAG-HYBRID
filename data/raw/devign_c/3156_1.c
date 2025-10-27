static int64_t asf_read_pts(AVFormatContext *s, int64_t *ppos, int stream_index)

{

    ASFContext *asf = s->priv_data;

    AVPacket pkt1, *pkt = &pkt1;

    int64_t pos= *ppos;

    int64_t pts;



    // ensure we are on the packet boundry

    assert(pos % asf->packet_size == 0);



    url_fseek(&s->pb, pos + s->data_offset, SEEK_SET);

    do{

        pos= url_ftell(&s->pb) - s->data_offset;

        asf_reset_header(s);

        if (av_read_frame(s, pkt) < 0)

    	    return AV_NOPTS_VALUE;

        pts= pkt->pts;



        av_free_packet(pkt);

    }while(pkt->stream_index != stream_index);

    

    *ppos= pos;



    return pts;

}
