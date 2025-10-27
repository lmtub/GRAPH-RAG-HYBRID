static int hls_start(AVFormatContext *s, VariantStream *vs)

{

    HLSContext *c = s->priv_data;

    AVFormatContext *oc = vs->avf;

    AVFormatContext *vtt_oc = vs->vtt_avf;

    AVDictionary *options = NULL;

    char *filename, iv_string[KEYSIZE*2 + 1];

    int err = 0;



    if (c->flags & HLS_SINGLE_FILE) {

        av_strlcpy(oc->filename, vs->basename,

                   sizeof(oc->filename));

        if (vs->vtt_basename)

            av_strlcpy(vtt_oc->filename, vs->vtt_basename,

                  sizeof(vtt_oc->filename));

    } else if (c->max_seg_size > 0) {

        if (replace_int_data_in_filename(oc->filename, sizeof(oc->filename),

#if FF_API_HLS_WRAP

            vs->basename, 'd', c->wrap ? vs->sequence % c->wrap : vs->sequence) < 1) {

#else

            vs->basename, 'd', vs->sequence) < 1) {

#endif

                av_log(oc, AV_LOG_ERROR, "Invalid segment filename template '%s', you can try to use -use_localtime 1 with it\n", vs->basename);

                return AVERROR(EINVAL);

        }

    } else {

        if (c->use_localtime) {

            time_t now0;

            struct tm *tm, tmpbuf;

            time(&now0);

            tm = localtime_r(&now0, &tmpbuf);

            if (!strftime(oc->filename, sizeof(oc->filename), vs->basename, tm)) {

                av_log(oc, AV_LOG_ERROR, "Could not get segment filename with use_localtime\n");

                return AVERROR(EINVAL);

            }



            err = sls_flag_use_localtime_filename(oc, c, vs);

            if (err < 0) {

                return AVERROR(ENOMEM);

            }



            if (c->use_localtime_mkdir) {

                const char *dir;

                char *fn_copy = av_strdup(oc->filename);

                if (!fn_copy) {

                    return AVERROR(ENOMEM);

                }

                dir = av_dirname(fn_copy);

                if (mkdir_p(dir) == -1 && errno != EEXIST) {

                    av_log(oc, AV_LOG_ERROR, "Could not create directory %s with use_localtime_mkdir\n", dir);

                    av_free(fn_copy);

                    return AVERROR(errno);

                }

                av_free(fn_copy);

            }

        } else if (replace_int_data_in_filename(oc->filename, sizeof(oc->filename),

#if FF_API_HLS_WRAP

                   vs->basename, 'd', c->wrap ? vs->sequence % c->wrap : vs->sequence) < 1) {

#else

                   vs->basename, 'd', vs->sequence) < 1) {

#endif

            av_log(oc, AV_LOG_ERROR, "Invalid segment filename template '%s' you can try to use -use_localtime 1 with it\n", vs->basename);

            return AVERROR(EINVAL);

        }

        if( vs->vtt_basename) {

            if (replace_int_data_in_filename(vtt_oc->filename, sizeof(vtt_oc->filename),

#if FF_API_HLS_WRAP

                vs->vtt_basename, 'd', c->wrap ? vs->sequence % c->wrap : vs->sequence) < 1) {

#else

                vs->vtt_basename, 'd', vs->sequence) < 1) {

#endif

                av_log(vtt_oc, AV_LOG_ERROR, "Invalid segment filename template '%s'\n", vs->vtt_basename);

                return AVERROR(EINVAL);

            }

       }

    }

    vs->number++;



    set_http_options(s, &options, c);



    if (c->flags & HLS_TEMP_FILE) {

        av_strlcat(oc->filename, ".tmp", sizeof(oc->filename));

    }



    if (c->key_info_file || c->encrypt) {

        if (c->key_info_file && c->encrypt) {

            av_log(s, AV_LOG_WARNING, "Cannot use both -hls_key_info_file and -hls_enc,"

                  " will use -hls_key_info_file priority\n");

        }



        if (!c->encrypt_started || (c->flags & HLS_PERIODIC_REKEY)) {

            if (c->key_info_file) {

                if ((err = hls_encryption_start(s)) < 0)

                    goto fail;

            } else {

                if ((err = do_encrypt(s, vs)) < 0)

                    goto fail;

            }

            c->encrypt_started = 1;

        }

        if ((err = av_dict_set(&options, "encryption_key", c->key_string, 0))

                < 0)

            goto fail;

        err = av_strlcpy(iv_string, c->iv_string, sizeof(iv_string));

        if (!err)

            snprintf(iv_string, sizeof(iv_string), "%032"PRIx64, vs->sequence);

        if ((err = av_dict_set(&options, "encryption_iv", iv_string, 0)) < 0)

           goto fail;



        filename = av_asprintf("crypto:%s", oc->filename);

        if (!filename) {

            err = AVERROR(ENOMEM);

            goto fail;

        }

        err = hlsenc_io_open(s, &oc->pb, filename, &options);

        av_free(filename);

        av_dict_free(&options);

        if (err < 0)

            return err;

    } else if (c->segment_type != SEGMENT_TYPE_FMP4) {

        if ((err = hlsenc_io_open(s, &oc->pb, oc->filename, &options)) < 0)

            goto fail;

    }

    if (vs->vtt_basename) {

        set_http_options(s, &options, c);

        if ((err = hlsenc_io_open(s, &vtt_oc->pb, vtt_oc->filename, &options)) < 0)

            goto fail;

    }

    av_dict_free(&options);



    if (c->segment_type != SEGMENT_TYPE_FMP4) {

        /* We only require one PAT/PMT per segment. */

        if (oc->oformat->priv_class && oc->priv_data) {

            char period[21];



            snprintf(period, sizeof(period), "%d", (INT_MAX / 2) - 1);



            av_opt_set(oc->priv_data, "mpegts_flags", "resend_headers", 0);

            av_opt_set(oc->priv_data, "sdt_period", period, 0);

            av_opt_set(oc->priv_data, "pat_period", period, 0);

        }

    }



    if (vs->vtt_basename) {

        err = avformat_write_header(vtt_oc,NULL);

        if (err < 0)

            return err;

    }



    return 0;

fail:

    av_dict_free(&options);



    return err;

}
