static int sls_flags_filename_process(struct AVFormatContext *s, HLSContext *hls,

                                      VariantStream *vs, HLSSegment *en,

                                      double duration, int64_t pos, int64_t size)

{

    if ((hls->flags & (HLS_SECOND_LEVEL_SEGMENT_SIZE | HLS_SECOND_LEVEL_SEGMENT_DURATION)) &&

        strlen(vs->current_segment_final_filename_fmt)) {

        av_strlcpy(vs->avf->filename, vs->current_segment_final_filename_fmt, sizeof(vs->avf->filename));

        if (hls->flags & HLS_SECOND_LEVEL_SEGMENT_SIZE) {

            char * filename = av_strdup(vs->avf->filename);  // %%s will be %s after strftime

            if (!filename) {

                av_free(en);

                return AVERROR(ENOMEM);

            }

            if (replace_int_data_in_filename(vs->avf->filename, sizeof(vs->avf->filename),

                filename, 's', pos + size) < 1) {

                av_log(hls, AV_LOG_ERROR,

                       "Invalid second level segment filename template '%s', "

                        "you can try to remove second_level_segment_size flag\n",

                       filename);

                av_free(filename);

                av_free(en);

                return AVERROR(EINVAL);

            }

            av_free(filename);

        }

        if (hls->flags & HLS_SECOND_LEVEL_SEGMENT_DURATION) {

            char * filename = av_strdup(vs->avf->filename);  // %%t will be %t after strftime

            if (!filename) {

                av_free(en);

                return AVERROR(ENOMEM);

            }

            if (replace_int_data_in_filename(vs->avf->filename, sizeof(vs->avf->filename),

                filename, 't',  (int64_t)round(duration * HLS_MICROSECOND_UNIT)) < 1) {

                av_log(hls, AV_LOG_ERROR,

                       "Invalid second level segment filename template '%s', "

                        "you can try to remove second_level_segment_time flag\n",

                       filename);

                av_free(filename);

                av_free(en);

                return AVERROR(EINVAL);

            }

            av_free(filename);

        }

    }

    return 0;

}
