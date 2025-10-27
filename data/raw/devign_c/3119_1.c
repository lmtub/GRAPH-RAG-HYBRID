static int seg_write_header(AVFormatContext *s)

{

    SegmentContext *seg = s->priv_data;

    AVFormatContext *oc = NULL;

    int ret, i;



    seg->segment_count = 0;

    if (!seg->write_header_trailer)

        seg->individual_header_trailer = 0;



    if (seg->time_str && seg->times_str) {

        av_log(s, AV_LOG_ERROR,

               "segment_time and segment_times options are mutually exclusive, select just one of them\n");

        return AVERROR(EINVAL);

    }



    if ((seg->list_flags & SEGMENT_LIST_FLAG_LIVE) && seg->times_str) {

        av_log(s, AV_LOG_ERROR,

               "segment_flags +live and segment_times options are mutually exclusive:"

               "specify -segment_time if you want a live-friendly list\n");

        return AVERROR(EINVAL);

    }



    if (seg->times_str) {

        if ((ret = parse_times(s, &seg->times, &seg->nb_times, seg->times_str)) < 0)

            return ret;

    } else {

        /* set default value if not specified */

        if (!seg->time_str)

            seg->time_str = av_strdup("2");

        if ((ret = av_parse_time(&seg->time, seg->time_str, 1)) < 0) {

            av_log(s, AV_LOG_ERROR,

                   "Invalid time duration specification '%s' for segment_time option\n",

                   seg->time_str);

            return ret;

        }

    }



    if (seg->time_delta_str) {

        if ((ret = av_parse_time(&seg->time_delta, seg->time_delta_str, 1)) < 0) {

            av_log(s, AV_LOG_ERROR,

                   "Invalid time duration specification '%s' for delta option\n",

                   seg->time_delta_str);

            return ret;

        }

    }



    if (seg->list) {

        if (seg->list_type == LIST_TYPE_UNDEFINED) {

            if      (av_match_ext(seg->list, "csv" )) seg->list_type = LIST_TYPE_CSV;

            else if (av_match_ext(seg->list, "ext" )) seg->list_type = LIST_TYPE_EXT;

            else if (av_match_ext(seg->list, "m3u8")) seg->list_type = LIST_TYPE_M3U8;

            else                                      seg->list_type = LIST_TYPE_FLAT;

        }

        if ((ret = segment_list_open(s)) < 0)

            goto fail;

    }

    if (seg->list_type == LIST_TYPE_EXT)

        av_log(s, AV_LOG_WARNING, "'ext' list type option is deprecated in favor of 'csv'\n");



    for (i = 0; i < s->nb_streams; i++)

        seg->has_video +=

            (s->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO);



    if (seg->has_video > 1)

        av_log(s, AV_LOG_WARNING,

               "More than a single video stream present, "

               "expect issues decoding it.\n");



    seg->oformat = av_guess_format(seg->format, s->filename, NULL);



    if (!seg->oformat) {

        ret = AVERROR_MUXER_NOT_FOUND;

        goto fail;

    }

    if (seg->oformat->flags & AVFMT_NOFILE) {

        av_log(s, AV_LOG_ERROR, "format %s not supported.\n",

               oc->oformat->name);

        ret = AVERROR(EINVAL);

        goto fail;

    }



    if ((ret = segment_mux_init(s)) < 0)

        goto fail;

    oc = seg->avf;



    if (av_get_frame_filename(oc->filename, sizeof(oc->filename),

                              s->filename, seg->segment_idx++) < 0) {

        ret = AVERROR(EINVAL);

        goto fail;

    }

    seg->segment_count++;



    if (seg->write_header_trailer) {

        if ((ret = avio_open2(&oc->pb, oc->filename, AVIO_FLAG_WRITE,

                              &s->interrupt_callback, NULL)) < 0)

            goto fail;

    } else {

        if ((ret = open_null_ctx(&oc->pb)) < 0)

            goto fail;

    }



    if ((ret = avformat_write_header(oc, NULL)) < 0) {

        avio_close(oc->pb);

        goto fail;

    }



    if (oc->avoid_negative_ts > 0 && s->avoid_negative_ts < 0)

        s->avoid_negative_ts = 1;



    if (!seg->write_header_trailer) {

        close_null_ctx(oc->pb);

        if ((ret = avio_open2(&oc->pb, oc->filename, AVIO_FLAG_WRITE,

                              &s->interrupt_callback, NULL)) < 0)

            goto fail;

    }



fail:

    if (ret) {

        if (seg->list)

            segment_list_close(s);

        if (seg->avf)

            avformat_free_context(seg->avf);

    }

    return ret;

}
