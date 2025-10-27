static int64_t asf_read_pts(AVFormatContext *s, int stream_index, int64_t *ppos, int64_t pos_limit)

{

    AVPacket pkt1, *pkt = &pkt1;

    ASFStream *asf_st;

    int64_t pts;

    int64_t pos= *ppos;

    int i;

    int64_t start_pos[ASF_MAX_STREAMS];



    for(i=0; i<s->nb_streams; i++){

        start_pos[i]= pos;

    }



    if (s->packet_size > 0)

        pos= (pos+s->packet_size-1-s->data_offset)/s->packet_size*s->packet_size+ s->data_offset;

    *ppos= pos;

    avio_seek(s->pb, pos, SEEK_SET);



//printf("asf_read_pts\n");

    asf_reset_header(s);

    for(;;){

        if (av_read_frame(s, pkt) < 0){

            av_log(s, AV_LOG_INFO, "asf_read_pts failed\n");

            return AV_NOPTS_VALUE;

        }



        pts= pkt->pts;



        av_free_packet(pkt);

        if(pkt->flags&AV_PKT_FLAG_KEY){

            i= pkt->stream_index;



            asf_st= s->streams[i]->priv_data;



//            assert((asf_st->packet_pos - s->data_offset) % s->packet_size == 0);

            pos= asf_st->packet_pos;



            av_add_index_entry(s->streams[i], pos, pts, pkt->size, pos - start_pos[i] + 1, AVINDEX_KEYFRAME);

            start_pos[i]= asf_st->packet_pos + 1;



            if(pkt->stream_index == stream_index)

               break;

        }

    }



    *ppos= pos;

//printf("found keyframe at %"PRId64" stream %d stamp:%"PRId64"\n", *ppos, stream_index, pts);



    return pts;

}
