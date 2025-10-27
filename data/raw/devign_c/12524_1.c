int av_seek_frame_binary(AVFormatContext *s, int stream_index, int64_t target_ts, int flags){

    AVInputFormat *avif= s->iformat;

    int64_t av_uninit(pos_min), av_uninit(pos_max), pos, pos_limit;

    int64_t ts_min, ts_max, ts;

    int index;

    AVStream *st;



    if (stream_index < 0)

        return -1;



#ifdef DEBUG_SEEK

    av_log(s, AV_LOG_DEBUG, "read_seek: %d %"PRId64"\n", stream_index, target_ts);

#endif



    ts_max=

    ts_min= AV_NOPTS_VALUE;

    pos_limit= -1; //gcc falsely says it may be uninitialized



    st= s->streams[stream_index];

    if(st->index_entries){

        AVIndexEntry *e;



        index= av_index_search_timestamp(st, target_ts, flags | AVSEEK_FLAG_BACKWARD); //FIXME whole func must be checked for non-keyframe entries in index case, especially read_timestamp()

        index= FFMAX(index, 0);

        e= &st->index_entries[index];



        if(e->timestamp <= target_ts || e->pos == e->min_distance){

            pos_min= e->pos;

            ts_min= e->timestamp;

#ifdef DEBUG_SEEK

            av_log(s, AV_LOG_DEBUG, "using cached pos_min=0x%"PRIx64" dts_min=%"PRId64"\n",

                   pos_min,ts_min);

#endif

        }else{

            assert(index==0);

        }



        index= av_index_search_timestamp(st, target_ts, flags & ~AVSEEK_FLAG_BACKWARD);

        assert(index < st->nb_index_entries);

        if(index >= 0){

            e= &st->index_entries[index];

            assert(e->timestamp >= target_ts);

            pos_max= e->pos;

            ts_max= e->timestamp;

            pos_limit= pos_max - e->min_distance;

#ifdef DEBUG_SEEK

            av_log(s, AV_LOG_DEBUG, "using cached pos_max=0x%"PRIx64" pos_limit=0x%"PRIx64" dts_max=%"PRId64"\n",

                   pos_max,pos_limit, ts_max);

#endif

        }

    }



    pos= av_gen_search(s, stream_index, target_ts, pos_min, pos_max, pos_limit, ts_min, ts_max, flags, &ts, avif->read_timestamp);

    if(pos<0)

        return -1;



    /* do the seek */

    url_fseek(s->pb, pos, SEEK_SET);



    av_update_cur_dts(s, st, ts);



    return 0;

}
