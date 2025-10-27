static int64_t rm_read_dts(AVFormatContext *s, int stream_index,

                               int64_t *ppos, int64_t pos_limit)

{

    RMDemuxContext *rm = s->priv_data;

    int64_t pos, dts;

    int stream_index2, flags, len, h;



    pos = *ppos;



    if(rm->old_format)

        return AV_NOPTS_VALUE;



    avio_seek(s->pb, pos, SEEK_SET);

    rm->remaining_len=0;

    for(;;){

        int seq=1;

        AVStream *st;



        len=sync(s, &dts, &flags, &stream_index2, &pos);

        if(len<0)

            return AV_NOPTS_VALUE;



        st = s->streams[stream_index2];

        if (st->codec->codec_type == AVMEDIA_TYPE_VIDEO) {

            h= avio_r8(s->pb); len--;

            if(!(h & 0x40)){

                seq = avio_r8(s->pb); len--;

            }

        }



        if((flags&2) && (seq&0x7F) == 1){

//            av_log(s, AV_LOG_DEBUG, "%d %d-%d %"PRId64" %d\n", flags, stream_index2, stream_index, dts, seq);

            av_add_index_entry(st, pos, dts, 0, 0, AVINDEX_KEYFRAME);

            if(stream_index2 == stream_index)

                break;

        }



        avio_skip(s->pb, len);

    }

    *ppos = pos;

    return dts;

}
