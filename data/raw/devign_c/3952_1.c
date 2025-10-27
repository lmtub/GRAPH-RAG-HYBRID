static int hls_append_segment(struct AVFormatContext *s, HLSContext *hls, double duration,

                              int64_t pos, int64_t size)

{

    HLSSegment *en = av_malloc(sizeof(*en));

    const char  *filename;

    int ret;



    if (!en)

        return AVERROR(ENOMEM);



    if ((hls->flags & (HLS_SECOND_LEVEL_SEGMENT_SIZE | HLS_SECOND_LEVEL_SEGMENT_DURATION)) &&

        strlen(hls->current_segment_final_filename_fmt)) {

        char * old_filename = av_strdup(hls->avf->filename);  // %%s will be %s after strftime

        av_strlcpy(hls->avf->filename, hls->current_segment_final_filename_fmt, sizeof(hls->avf->filename));

        if (hls->flags & HLS_SECOND_LEVEL_SEGMENT_SIZE) {

            char * filename = av_strdup(hls->avf->filename);  // %%s will be %s after strftime

            if (!filename)

                return AVERROR(ENOMEM);

            if (replace_int_data_in_filename(hls->avf->filename, sizeof(hls->avf->filename),

                filename, 's', pos + size) < 1) {

                av_log(hls, AV_LOG_ERROR,

                       "Invalid second level segment filename template '%s', "

                        "you can try to remove second_level_segment_size flag\n",

                       filename);

                av_free(filename);

                av_free(old_filename);

                return AVERROR(EINVAL);

            }

            av_free(filename);

        }

        if (hls->flags & HLS_SECOND_LEVEL_SEGMENT_DURATION) {

            char * filename = av_strdup(hls->avf->filename);  // %%t will be %t after strftime

            if (!filename)

                return AVERROR(ENOMEM);

            if (replace_int_data_in_filename(hls->avf->filename, sizeof(hls->avf->filename),

                filename, 't',  (int64_t)round(1000000 * duration)) < 1) {

                av_log(hls, AV_LOG_ERROR,

                       "Invalid second level segment filename template '%s', "

                        "you can try to remove second_level_segment_time flag\n",

                       filename);

                av_free(filename);

                av_free(old_filename);

                return AVERROR(EINVAL);

            }

            av_free(filename);

        }

        ff_rename(old_filename, hls->avf->filename, hls);

        av_free(old_filename);

    }





    filename = av_basename(hls->avf->filename);



    if (hls->use_localtime_mkdir) {

        filename = hls->avf->filename;

    }

    if (find_segment_by_filename(hls->segments, filename)

        || find_segment_by_filename(hls->old_segments, filename)) {

        av_log(hls, AV_LOG_WARNING, "Duplicated segment filename detected: %s\n", filename);

    }

    av_strlcpy(en->filename, filename, sizeof(en->filename));



    if(hls->has_subtitle)

        av_strlcpy(en->sub_filename, av_basename(hls->vtt_avf->filename), sizeof(en->sub_filename));

    else

        en->sub_filename[0] = '\0';



    en->duration = duration;

    en->pos      = pos;

    en->size     = size;

    en->next     = NULL;

    en->discont  = 0;



    if (hls->discontinuity) {

        en->discont = 1;

        hls->discontinuity = 0;

    }



    if (hls->key_info_file) {

        av_strlcpy(en->key_uri, hls->key_uri, sizeof(en->key_uri));

        av_strlcpy(en->iv_string, hls->iv_string, sizeof(en->iv_string));

    }



    if (!hls->segments)

        hls->segments = en;

    else

        hls->last_segment->next = en;



    hls->last_segment = en;



    // EVENT or VOD playlists imply sliding window cannot be used

    if (hls->pl_type != PLAYLIST_TYPE_NONE)

        hls->max_nb_segments = 0;



    if (hls->max_nb_segments && hls->nb_entries >= hls->max_nb_segments) {

        en = hls->segments;

        hls->initial_prog_date_time += en->duration;

        hls->segments = en->next;

        if (en && hls->flags & HLS_DELETE_SEGMENTS &&

                !(hls->flags & HLS_SINGLE_FILE || hls->wrap)) {

            en->next = hls->old_segments;

            hls->old_segments = en;

            if ((ret = hls_delete_old_segments(hls)) < 0)

                return ret;

        } else

            av_free(en);

    } else

        hls->nb_entries++;



    if (hls->max_seg_size > 0) {

        return 0;

    }

    hls->sequence++;



    return 0;

}
