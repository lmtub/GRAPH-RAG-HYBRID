static int read_seek(AVFormatContext *s, int stream_index,

                     int64_t pts, int flags)

{

    NUTContext *nut    = s->priv_data;

    AVStream *st       = s->streams[stream_index];

    Syncpoint dummy    = { .ts = pts * av_q2d(st->time_base) * AV_TIME_BASE };

    Syncpoint nopts_sp = { .ts = AV_NOPTS_VALUE, .back_ptr = AV_NOPTS_VALUE };

    Syncpoint *sp, *next_node[2] = { &nopts_sp, &nopts_sp };

    int64_t pos, pos2, ts;

    int i;



    if (nut->flags & NUT_PIPE) {

        return AVERROR(ENOSYS);

    }



    if (st->index_entries) {

        int index = av_index_search_timestamp(st, pts, flags);

        if (index < 0)

            index = av_index_search_timestamp(st, pts, flags ^ AVSEEK_FLAG_BACKWARD);

        if (index < 0)

            return -1;



        pos2 = st->index_entries[index].pos;

        ts   = st->index_entries[index].timestamp;

    } else {

        av_tree_find(nut->syncpoints, &dummy, (void *) ff_nut_sp_pts_cmp,

                     (void **) next_node);

        av_log(s, AV_LOG_DEBUG, "%"PRIu64"-%"PRIu64" %"PRId64"-%"PRId64"\n",

               next_node[0]->pos, next_node[1]->pos, next_node[0]->ts,

               next_node[1]->ts);

        pos = ff_gen_search(s, -1, dummy.ts, next_node[0]->pos,

                            next_node[1]->pos, next_node[1]->pos,

                            next_node[0]->ts, next_node[1]->ts,

                            AVSEEK_FLAG_BACKWARD, &ts, nut_read_timestamp);





        if (!(flags & AVSEEK_FLAG_BACKWARD)) {

            dummy.pos    = pos + 16;

            next_node[1] = &nopts_sp;

            av_tree_find(nut->syncpoints, &dummy, (void *) ff_nut_sp_pos_cmp,

                         (void **) next_node);

            pos2 = ff_gen_search(s, -2, dummy.pos, next_node[0]->pos,

                                 next_node[1]->pos, next_node[1]->pos,

                                 next_node[0]->back_ptr, next_node[1]->back_ptr,

                                 flags, &ts, nut_read_timestamp);

            if (pos2 >= 0)

                pos = pos2;

            // FIXME dir but I think it does not matter

        }

        dummy.pos = pos;

        sp = av_tree_find(nut->syncpoints, &dummy, (void *) ff_nut_sp_pos_cmp,

                          NULL);



        av_assert0(sp);

        pos2 = sp->back_ptr - 15;

    }

    av_log(NULL, AV_LOG_DEBUG, "SEEKTO: %"PRId64"\n", pos2);

    pos = find_startcode(s->pb, SYNCPOINT_STARTCODE, pos2);

    avio_seek(s->pb, pos, SEEK_SET);

    nut->last_syncpoint_pos = pos;

    av_log(NULL, AV_LOG_DEBUG, "SP: %"PRId64"\n", pos);

    if (pos2 > pos || pos2 + 15 < pos)

        av_log(NULL, AV_LOG_ERROR, "no syncpoint at backptr pos\n");

    for (i = 0; i < s->nb_streams; i++)

        nut->stream[i].skip_until_key_frame = 1;



    nut->last_resync_pos = 0;



    return 0;

}