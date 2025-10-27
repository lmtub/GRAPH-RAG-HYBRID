static int rm_read_packet(AVFormatContext *s, AVPacket *pkt)

{

    RMDemuxContext *rm = s->priv_data;

    AVStream *st;

    int i, len, res, seq = 1;

    int64_t timestamp, pos;

    int flags;



    for (;;) {

        if (rm->audio_pkt_cnt) {

            // If there are queued audio packet return them first

            st = s->streams[rm->audio_stream_num];

            ff_rm_retrieve_cache(s, s->pb, st, st->priv_data, pkt);

            flags = 0;

        } else {

            if (rm->old_format) {

                RMStream *ast;



                st = s->streams[0];

                ast = st->priv_data;

                timestamp = AV_NOPTS_VALUE;

                len = !ast->audio_framesize ? RAW_PACKET_SIZE :

                    ast->coded_framesize * ast->sub_packet_h / 2;

                flags = (seq++ == 1) ? 2 : 0;

                pos = avio_tell(s->pb);

            } else {

                len=sync(s, &timestamp, &flags, &i, &pos);

                if (len > 0)

                    st = s->streams[i];

            }



            if(len<0 || s->pb->eof_reached)

                return AVERROR(EIO);



            res = ff_rm_parse_packet (s, s->pb, st, st->priv_data, len, pkt,

                                      &seq, flags, timestamp);

            if((flags&2) && (seq&0x7F) == 1)

                av_add_index_entry(st, pos, timestamp, 0, 0, AVINDEX_KEYFRAME);

            if (res)

                continue;

        }



        if(  (st->discard >= AVDISCARD_NONKEY && !(flags&2))

           || st->discard >= AVDISCARD_ALL){

            av_free_packet(pkt);

        } else

            break;

    }



    return 0;

}
