static void output_segment_list(OutputStream *os, AVIOContext *out, DASHContext *c,

                                int representation_id, int final)

{

    int i, start_index = 0, start_number = 1;

    if (c->window_size) {

        start_index  = FFMAX(os->nb_segments   - c->window_size, 0);

        start_number = FFMAX(os->segment_index - c->window_size, 1);

    }



    if (c->use_template) {

        int timescale = c->use_timeline ? os->ctx->streams[0]->time_base.den : AV_TIME_BASE;

        avio_printf(out, "\t\t\t\t<SegmentTemplate timescale=\"%d\" ", timescale);

        if (!c->use_timeline)

            avio_printf(out, "duration=\"%"PRId64"\" ", c->last_duration);

        avio_printf(out, "initialization=\"%s\" media=\"%s\" startNumber=\"%d\">\n", c->init_seg_name, c->media_seg_name, c->use_timeline ? start_number : 1);

        if (c->use_timeline) {

            int64_t cur_time = 0;

            avio_printf(out, "\t\t\t\t\t<SegmentTimeline>\n");

            for (i = start_index; i < os->nb_segments; ) {

                Segment *seg = os->segments[i];

                int repeat = 0;

                avio_printf(out, "\t\t\t\t\t\t<S ");

                if (i == start_index || seg->time != cur_time) {

                    cur_time = seg->time;

                    avio_printf(out, "t=\"%"PRId64"\" ", seg->time);

                }

                avio_printf(out, "d=\"%d\" ", seg->duration);

                while (i + repeat + 1 < os->nb_segments &&

                       os->segments[i + repeat + 1]->duration == seg->duration &&

                       os->segments[i + repeat + 1]->time == os->segments[i + repeat]->time + os->segments[i + repeat]->duration)

                    repeat++;

                if (repeat > 0)

                    avio_printf(out, "r=\"%d\" ", repeat);

                avio_printf(out, "/>\n");

                i += 1 + repeat;

                cur_time += (1 + repeat) * seg->duration;

            }

            avio_printf(out, "\t\t\t\t\t</SegmentTimeline>\n");

        }

        avio_printf(out, "\t\t\t\t</SegmentTemplate>\n");

    } else if (c->single_file) {

        avio_printf(out, "\t\t\t\t<BaseURL>%s</BaseURL>\n", os->initfile);

        avio_printf(out, "\t\t\t\t<SegmentList timescale=\"%d\" duration=\"%"PRId64"\" startNumber=\"%d\">\n", AV_TIME_BASE, c->last_duration, start_number);

        avio_printf(out, "\t\t\t\t\t<Initialization range=\"%"PRId64"-%"PRId64"\" />\n", os->init_start_pos, os->init_start_pos + os->init_range_length - 1);

        for (i = start_index; i < os->nb_segments; i++) {

            Segment *seg = os->segments[i];

            avio_printf(out, "\t\t\t\t\t<SegmentURL mediaRange=\"%"PRId64"-%"PRId64"\" ", seg->start_pos, seg->start_pos + seg->range_length - 1);

            if (seg->index_length)

                avio_printf(out, "indexRange=\"%"PRId64"-%"PRId64"\" ", seg->start_pos, seg->start_pos + seg->index_length - 1);

            avio_printf(out, "/>\n");

        }

        avio_printf(out, "\t\t\t\t</SegmentList>\n");

    } else {

        avio_printf(out, "\t\t\t\t<SegmentList timescale=\"%d\" duration=\"%"PRId64"\" startNumber=\"%d\">\n", AV_TIME_BASE, c->last_duration, start_number);

        avio_printf(out, "\t\t\t\t\t<Initialization sourceURL=\"%s\" />\n", os->initfile);

        for (i = start_index; i < os->nb_segments; i++) {

            Segment *seg = os->segments[i];

            avio_printf(out, "\t\t\t\t\t<SegmentURL media=\"%s\" />\n", seg->file);

        }

        avio_printf(out, "\t\t\t\t</SegmentList>\n");

    }

    if (c->hls_playlist && start_index < os->nb_segments)

    {

        int timescale = os->ctx->streams[0]->time_base.den;

        char temp_filename_hls[1024];

        char filename_hls[1024];

        AVIOContext *out_hls = NULL;

        AVDictionary *http_opts = NULL;

        int target_duration = 0;

        int ret = 0;

        const char *proto = avio_find_protocol_name(c->dirname);

        int use_rename = proto && !strcmp(proto, "file");



        get_hls_playlist_name(filename_hls, sizeof(filename_hls),

                              c->dirname, representation_id);



        snprintf(temp_filename_hls, sizeof(temp_filename_hls), use_rename ? "%s.tmp" : "%s", filename_hls);



        set_http_options(&http_opts, c);

        avio_open2(&out_hls, temp_filename_hls, AVIO_FLAG_WRITE, NULL, &http_opts);

        av_dict_free(&http_opts);

        for (i = start_index; i < os->nb_segments; i++) {

            Segment *seg = os->segments[i];

            double duration = (double) seg->duration / timescale;

            if (target_duration <= duration)

                target_duration = hls_get_int_from_double(duration);

        }



        ff_hls_write_playlist_header(out_hls, 6, -1, target_duration,

                                     start_number, PLAYLIST_TYPE_NONE);



        ff_hls_write_init_file(out_hls, os->initfile, c->single_file,

                               os->init_range_length, os->init_start_pos);



        for (i = start_index; i < os->nb_segments; i++) {

            Segment *seg = os->segments[i];

            ret = ff_hls_write_file_entry(out_hls, 0, c->single_file,

                                    (double) seg->duration / timescale, 0,

                                    seg->range_length, seg->start_pos, NULL,

                                    c->single_file ? os->initfile : seg->file,

                                    NULL);

            if (ret < 0) {

                av_log(os->ctx, AV_LOG_WARNING, "ff_hls_write_file_entry get error\n");

            }

        }



        if (final)

            ff_hls_write_end_list(out_hls);



        avio_close(out_hls);

        if (use_rename)

            avpriv_io_move(temp_filename_hls, filename_hls);

    }



}
