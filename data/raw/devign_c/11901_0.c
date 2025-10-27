static int hls_write_header(AVFormatContext *s)

{

    HLSContext *hls = s->priv_data;

    int ret, i;

    char *p;

    const char *pattern = "%d.ts";

    const char *pattern_localtime_fmt = "-%s.ts";

    const char *vtt_pattern = "%d.vtt";

    AVDictionary *options = NULL;

    int basename_size;

    int vtt_basename_size;



    hls->sequence       = hls->start_sequence;

    hls->recording_time = (hls->init_time ? hls->init_time : hls->time) * AV_TIME_BASE;

    hls->start_pts      = AV_NOPTS_VALUE;



    if (hls->flags & HLS_PROGRAM_DATE_TIME) {

        time_t now0;

        time(&now0);

        hls->initial_prog_date_time = now0;

    }



    if (hls->format_options_str) {

        ret = av_dict_parse_string(&hls->format_options, hls->format_options_str, "=", ":", 0);

        if (ret < 0) {

            av_log(s, AV_LOG_ERROR, "Could not parse format options list '%s'\n", hls->format_options_str);

            goto fail;

        }

    }



    for (i = 0; i < s->nb_streams; i++) {

        hls->has_video +=

            s->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO;

        hls->has_subtitle +=

            s->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_SUBTITLE;

    }



    if (hls->has_video > 1)

        av_log(s, AV_LOG_WARNING,

               "More than a single video stream present, "

               "expect issues decoding it.\n");



    hls->oformat = av_guess_format("mpegts", NULL, NULL);



    if (!hls->oformat) {

        ret = AVERROR_MUXER_NOT_FOUND;

        goto fail;

    }



    if(hls->has_subtitle) {

        hls->vtt_oformat = av_guess_format("webvtt", NULL, NULL);

        if (!hls->oformat) {

            ret = AVERROR_MUXER_NOT_FOUND;

            goto fail;

        }

    }



    if (hls->segment_filename) {

        hls->basename = av_strdup(hls->segment_filename);

        if (!hls->basename) {

            ret = AVERROR(ENOMEM);

            goto fail;

        }

    } else {

        if (hls->flags & HLS_SINGLE_FILE)

            pattern = ".ts";



        if (hls->use_localtime) {

            basename_size = strlen(s->filename) + strlen(pattern_localtime_fmt) + 1;

        } else {

            basename_size = strlen(s->filename) + strlen(pattern) + 1;

        }

        hls->basename = av_malloc(basename_size);

        if (!hls->basename) {

            ret = AVERROR(ENOMEM);

            goto fail;

        }



        av_strlcpy(hls->basename, s->filename, basename_size);



        p = strrchr(hls->basename, '.');

        if (p)

            *p = '\0';

        if (hls->use_localtime) {

            av_strlcat(hls->basename, pattern_localtime_fmt, basename_size);

        } else {

            av_strlcat(hls->basename, pattern, basename_size);

        }

    }

    if (!hls->use_localtime && (hls->flags & HLS_SECOND_LEVEL_SEGMENT_INDEX)) {

        av_log(hls, AV_LOG_ERROR, "second_level_segment_index hls_flag requires use_localtime to be true\n");

        ret = AVERROR(EINVAL);

        goto fail;

    }

    if(hls->has_subtitle) {



        if (hls->flags & HLS_SINGLE_FILE)

            vtt_pattern = ".vtt";

        vtt_basename_size = strlen(s->filename) + strlen(vtt_pattern) + 1;

        hls->vtt_basename = av_malloc(vtt_basename_size);

        if (!hls->vtt_basename) {

            ret = AVERROR(ENOMEM);

            goto fail;

        }

        hls->vtt_m3u8_name = av_malloc(vtt_basename_size);

        if (!hls->vtt_m3u8_name ) {

            ret = AVERROR(ENOMEM);

            goto fail;

        }

        av_strlcpy(hls->vtt_basename, s->filename, vtt_basename_size);

        p = strrchr(hls->vtt_basename, '.');

        if (p)

            *p = '\0';



        if( hls->subtitle_filename ) {

            strcpy(hls->vtt_m3u8_name, hls->subtitle_filename);

        } else {

            strcpy(hls->vtt_m3u8_name, hls->vtt_basename);

            av_strlcat(hls->vtt_m3u8_name, "_vtt.m3u8", vtt_basename_size);

        }

        av_strlcat(hls->vtt_basename, vtt_pattern, vtt_basename_size);

    }



    if ((ret = hls_mux_init(s)) < 0)

        goto fail;



    if (hls->flags & HLS_APPEND_LIST) {

        parse_playlist(s, s->filename);

        hls->discontinuity = 1;

        if (hls->init_time > 0) {

            av_log(s, AV_LOG_WARNING, "append_list mode does not support hls_init_time,"

                   " hls_init_time value will have no effect\n");

            hls->init_time = 0;

            hls->recording_time = hls->time * AV_TIME_BASE;

        }

    }



    if ((ret = hls_start(s)) < 0)

        goto fail;



    av_dict_copy(&options, hls->format_options, 0);

    ret = avformat_write_header(hls->avf, &options);

    if (av_dict_count(options)) {

        av_log(s, AV_LOG_ERROR, "Some of provided format options in '%s' are not recognized\n", hls->format_options_str);

        ret = AVERROR(EINVAL);

        goto fail;

    }

    //av_assert0(s->nb_streams == hls->avf->nb_streams);

    for (i = 0; i < s->nb_streams; i++) {

        AVStream *inner_st;

        AVStream *outer_st = s->streams[i];



        if (hls->max_seg_size > 0) {

            if ((outer_st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) &&

                (outer_st->codecpar->bit_rate > hls->max_seg_size)) {

                av_log(s, AV_LOG_WARNING, "Your video bitrate is bigger than hls_segment_size, "

                       "(%"PRId64 " > %"PRId64 "), the result maybe not be what you want.",

                       outer_st->codecpar->bit_rate, hls->max_seg_size);

            }

        }



        if (outer_st->codecpar->codec_type != AVMEDIA_TYPE_SUBTITLE)

            inner_st = hls->avf->streams[i];

        else if (hls->vtt_avf)

            inner_st = hls->vtt_avf->streams[0];

        else {

            /* We have a subtitle stream, when the user does not want one */

            inner_st = NULL;

            continue;

        }

        avpriv_set_pts_info(outer_st, inner_st->pts_wrap_bits, inner_st->time_base.num, inner_st->time_base.den);

    }

fail:



    av_dict_free(&options);

    if (ret < 0) {

        av_freep(&hls->basename);

        av_freep(&hls->vtt_basename);

        if (hls->avf)

            avformat_free_context(hls->avf);

        if (hls->vtt_avf)

            avformat_free_context(hls->vtt_avf);



    }

    return ret;

}
