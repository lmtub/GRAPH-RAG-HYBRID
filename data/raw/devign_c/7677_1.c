int av_interleave_packet_per_dts(AVFormatContext *s, AVPacket *out, AVPacket *pkt, int flush){

    AVPacketList *pktl;

    int stream_count=0, noninterleaved_count=0;

    int64_t delta_dts_max = 0;

    int i;



    if(pkt){

        ff_interleave_add_packet(s, pkt, ff_interleave_compare_dts);

    }



    for(i=0; i < s->nb_streams; i++) {

        if (s->streams[i]->last_in_packet_buffer) {

            ++stream_count;

        } else if(s->streams[i]->codec->codec_type == AVMEDIA_TYPE_SUBTITLE) {

            ++noninterleaved_count;

        }

    }



    if (s->nb_streams == stream_count) {

        flush = 1;

    } else if (!flush){

        for(i=0; i < s->nb_streams; i++) {

            if (s->streams[i]->last_in_packet_buffer) {

                int64_t delta_dts =

                    av_rescale_q(s->streams[i]->last_in_packet_buffer->pkt.dts,

                                s->streams[i]->time_base,

                                AV_TIME_BASE_Q) -

                    av_rescale_q(s->packet_buffer->pkt.dts,

                                s->streams[s->packet_buffer->pkt.stream_index]->time_base,

                                AV_TIME_BASE_Q);

                delta_dts_max= FFMAX(delta_dts_max, delta_dts);

            }

        }

        if(s->nb_streams == stream_count+noninterleaved_count &&

           delta_dts_max > 20*AV_TIME_BASE) {

            av_log(s, AV_LOG_DEBUG, "flushing with %d noninterleaved\n", noninterleaved_count);

            flush = 1;

        }

    }

    if(stream_count && flush){

        pktl= s->packet_buffer;

        *out= pktl->pkt;



        s->packet_buffer= pktl->next;

        if(!s->packet_buffer)

            s->packet_buffer_end= NULL;



        if(s->streams[out->stream_index]->last_in_packet_buffer == pktl)

            s->streams[out->stream_index]->last_in_packet_buffer= NULL;

        av_freep(&pktl);

        return 1;

    }else{

        av_init_packet(out);

        return 0;

    }

}
