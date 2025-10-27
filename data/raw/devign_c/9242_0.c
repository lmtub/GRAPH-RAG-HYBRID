static int read_data(void *opaque, uint8_t *buf, int buf_size)

{

    struct playlist *v = opaque;

    HLSContext *c = v->parent->priv_data;

    int ret, i;

    int just_opened = 0;



    if (!v->needed)

        return AVERROR_EOF;



restart:

    if (!v->input) {

        /* If this is a live stream and the reload interval has elapsed since

         * the last playlist reload, reload the playlists now. */

        int64_t reload_interval = default_reload_interval(v);



reload:

        if (!v->finished &&

            av_gettime() - v->last_load_time >= reload_interval) {

            if ((ret = parse_playlist(c, v->url, v, NULL)) < 0) {

                av_log(v->parent, AV_LOG_WARNING, "Failed to reload playlist %d\n",

                       v->index);

                return ret;

            }

            /* If we need to reload the playlist again below (if

             * there's still no more segments), switch to a reload

             * interval of half the target duration. */

            reload_interval = v->target_duration / 2;

        }

        if (v->cur_seq_no < v->start_seq_no) {

            av_log(NULL, AV_LOG_WARNING,

                   "skipping %d segments ahead, expired from playlists\n",

                   v->start_seq_no - v->cur_seq_no);

            v->cur_seq_no = v->start_seq_no;

        }

        if (v->cur_seq_no >= v->start_seq_no + v->n_segments) {

            if (v->finished)

                return AVERROR_EOF;

            while (av_gettime() - v->last_load_time < reload_interval) {

                if (ff_check_interrupt(c->interrupt_callback))

                    return AVERROR_EXIT;

                av_usleep(100*1000);

            }

            /* Enough time has elapsed since the last reload */

            goto reload;

        }



        ret = open_input(c, v);

        if (ret < 0) {

            av_log(v->parent, AV_LOG_WARNING, "Failed to open segment of playlist %d\n",

                   v->index);

            return ret;

        }

        just_opened = 1;

    }



    ret = read_from_url(v, buf, buf_size, READ_NORMAL);

    if (ret > 0) {

        if (just_opened && v->is_id3_timestamped != 0) {

            /* Intercept ID3 tags here, elementary audio streams are required

             * to convey timestamps using them in the beginning of each segment. */

            intercept_id3(v, buf, buf_size, &ret);

        }



        return ret;

    }

    ffurl_close(v->input);

    v->input = NULL;

    v->cur_seq_no++;



    c->end_of_segment = 1;

    c->cur_seq_no = v->cur_seq_no;



    if (v->ctx && v->ctx->nb_streams &&

        v->parent->nb_streams >= v->stream_offset + v->ctx->nb_streams) {

        v->needed = 0;

        for (i = v->stream_offset; i < v->stream_offset + v->ctx->nb_streams;

             i++) {

            if (v->parent->streams[i]->discard < AVDISCARD_ALL)

                v->needed = 1;

        }

    }

    if (!v->needed) {

        av_log(v->parent, AV_LOG_INFO, "No longer receiving playlist %d\n",

               v->index);

        return AVERROR_EOF;

    }

    goto restart;

}
