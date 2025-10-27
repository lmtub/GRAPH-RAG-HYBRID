int av_probe_input_buffer2(AVIOContext *pb, AVInputFormat **fmt,

                          const char *filename, void *logctx,

                          unsigned int offset, unsigned int max_probe_size)

{

    AVProbeData pd = { filename ? filename : "" };

    uint8_t *buf = NULL;

    uint8_t *mime_type;

    int ret = 0, probe_size, buf_offset = 0;

    int score = 0;



    if (!max_probe_size)

        max_probe_size = PROBE_BUF_MAX;

    else if (max_probe_size < PROBE_BUF_MIN) {

        av_log(logctx, AV_LOG_ERROR,

               "Specified probe size value %u cannot be < %u\n", max_probe_size, PROBE_BUF_MIN);

        return AVERROR(EINVAL);

    }



    if (offset >= max_probe_size)

        return AVERROR(EINVAL);



#ifdef FF_API_PROBE_MIME

    if (pb->av_class)

        av_opt_get(pb, "mime_type", AV_OPT_SEARCH_CHILDREN, &pd.mime_type);

#endif



#if !FF_API_PROBE_MIME

    if (!*fmt && pb->av_class && av_opt_get(pb, "mime_type", AV_OPT_SEARCH_CHILDREN, &mime_type) >= 0 && mime_type) {

        if (!av_strcasecmp(mime_type, "audio/aacp")) {

            *fmt = av_find_input_format("aac");

        }

        av_freep(&mime_type);

    }

#endif



    for (probe_size = PROBE_BUF_MIN; probe_size <= max_probe_size && !*fmt;

         probe_size = FFMIN(probe_size << 1,

                            FFMAX(max_probe_size, probe_size + 1))) {

        score = probe_size < max_probe_size ? AVPROBE_SCORE_RETRY : 0;



        /* Read probe data. */

        if ((ret = av_reallocp(&buf, probe_size + AVPROBE_PADDING_SIZE)) < 0)

            goto fail;

        if ((ret = avio_read(pb, buf + buf_offset,

                             probe_size - buf_offset)) < 0) {

            /* Fail if error was not end of file, otherwise, lower score. */

            if (ret != AVERROR_EOF)

                goto fail;



            score = 0;

            ret   = 0;          /* error was end of file, nothing read */

        }

        buf_offset += ret;

        if (buf_offset < offset)

            continue;

        pd.buf_size = buf_offset - offset;

        pd.buf = &buf[offset];



        memset(pd.buf + pd.buf_size, 0, AVPROBE_PADDING_SIZE);



        /* Guess file format. */

        *fmt = av_probe_input_format2(&pd, 1, &score);

        if (*fmt) {

            /* This can only be true in the last iteration. */

            if (score <= AVPROBE_SCORE_RETRY) {

                av_log(logctx, AV_LOG_WARNING,

                       "Format %s detected only with low score of %d, "

                       "misdetection possible!\n", (*fmt)->name, score);

            } else

                av_log(logctx, AV_LOG_DEBUG,

                       "Format %s probed with size=%d and score=%d\n",

                       (*fmt)->name, probe_size, score);

#if 0

            FILE *f = fopen("probestat.tmp", "ab");

            fprintf(f, "probe_size:%d format:%s score:%d filename:%s\n", probe_size, (*fmt)->name, score, filename);

            fclose(f);

#endif

        }

    }



    if (!*fmt)

        ret = AVERROR_INVALIDDATA;



fail:

    /* Rewind. Reuse probe buffer to avoid seeking. */

    if (ret >= 0)

        ret = ffio_rewind_with_probe_data(pb, &buf, buf_offset);



#ifdef FF_API_PROBE_MIME

    av_free(pd.mime_type);

#endif

    return ret < 0 ? ret : score;

}
