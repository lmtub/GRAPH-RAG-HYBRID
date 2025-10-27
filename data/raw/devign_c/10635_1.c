static int avi_read_seek(AVFormatContext *s, int stream_index,

                         int64_t timestamp, int flags)

{

    AVIContext *avi = s->priv_data;

    AVStream *st;

    int i, index;

    int64_t pos, pos_min;

    AVIStream *ast;



    /* Does not matter which stream is requested dv in avi has the

     * stream information in the first video stream.

     */

    if (avi->dv_demux)

        stream_index = 0;



    if (!avi->index_loaded) {

        /* we only load the index on demand */

        avi_load_index(s);

        avi->index_loaded |= 1;

    }

    av_assert0(stream_index >= 0);



    st    = s->streams[stream_index];

    ast   = st->priv_data;

    index = av_index_search_timestamp(st,

                                      timestamp * FFMAX(ast->sample_size, 1),

                                      flags);

    if (index < 0) {

        if (st->nb_index_entries > 0)

            av_log(s, AV_LOG_DEBUG, "Failed to find timestamp %"PRId64 " in index %"PRId64 " .. %"PRId64 "\n",

                   timestamp * FFMAX(ast->sample_size, 1),

                   st->index_entries[0].timestamp,

                   st->index_entries[st->nb_index_entries - 1].timestamp);

        return AVERROR_INVALIDDATA;

    }



    /* find the position */

    pos       = st->index_entries[index].pos;

    timestamp = st->index_entries[index].timestamp / FFMAX(ast->sample_size, 1);



    av_log(s, AV_LOG_TRACE, "XX %"PRId64" %d %"PRId64"\n",

            timestamp, index, st->index_entries[index].timestamp);



    if (CONFIG_DV_DEMUXER && avi->dv_demux) {

        /* One and only one real stream for DV in AVI, and it has video  */

        /* offsets. Calling with other stream indexes should have failed */

        /* the av_index_search_timestamp call above.                     */



        if (avio_seek(s->pb, pos, SEEK_SET) < 0)

            return -1;



        /* Feed the DV video stream version of the timestamp to the */

        /* DV demux so it can synthesize correct timestamps.        */

        ff_dv_offset_reset(avi->dv_demux, timestamp);



        avi->stream_index = -1;

        return 0;

    }



    pos_min = pos;

    for (i = 0; i < s->nb_streams; i++) {

        AVStream *st2   = s->streams[i];

        AVIStream *ast2 = st2->priv_data;



        ast2->packet_size =

        ast2->remaining   = 0;



        if (ast2->sub_ctx) {

            seek_subtitle(st, st2, timestamp);

            continue;

        }



        if (st2->nb_index_entries <= 0)

            continue;



//        av_assert1(st2->codecpar->block_align);

        av_assert0(fabs(av_q2d(st2->time_base) - ast2->scale / (double)ast2->rate) < av_q2d(st2->time_base) * 0.00000001);

        index = av_index_search_timestamp(st2,

                                          av_rescale_q(timestamp,

                                                       st->time_base,

                                                       st2->time_base) *

                                          FFMAX(ast2->sample_size, 1),

                                          flags |

                                          AVSEEK_FLAG_BACKWARD |

                                          (st2->codecpar->codec_type != AVMEDIA_TYPE_VIDEO ? AVSEEK_FLAG_ANY : 0));

        if (index < 0)

            index = 0;

        ast2->seek_pos = st2->index_entries[index].pos;

        pos_min = FFMIN(pos_min,ast2->seek_pos);

    }

    for (i = 0; i < s->nb_streams; i++) {

        AVStream *st2 = s->streams[i];

        AVIStream *ast2 = st2->priv_data;



        if (ast2->sub_ctx || st2->nb_index_entries <= 0)

            continue;



        index = av_index_search_timestamp(

                st2,

                av_rescale_q(timestamp, st->time_base, st2->time_base) * FFMAX(ast2->sample_size, 1),

                flags | AVSEEK_FLAG_BACKWARD | (st2->codecpar->codec_type != AVMEDIA_TYPE_VIDEO ? AVSEEK_FLAG_ANY : 0));

        if (index < 0)

            index = 0;

        while (!avi->non_interleaved && index>0 && st2->index_entries[index-1].pos >= pos_min)

            index--;

        ast2->frame_offset = st2->index_entries[index].timestamp;

    }



    /* do the seek */

    if (avio_seek(s->pb, pos_min, SEEK_SET) < 0) {

        av_log(s, AV_LOG_ERROR, "Seek failed\n");

        return -1;

    }

    avi->stream_index = -1;

    avi->dts_max      = INT_MIN;

    return 0;

}
