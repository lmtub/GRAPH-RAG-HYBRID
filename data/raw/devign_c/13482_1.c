int ff_probe_input_buffer(ByteIOContext **pb, AVInputFormat **fmt,

                          const char *filename, void *logctx,

                          unsigned int offset, unsigned int max_probe_size)

{

    AVProbeData pd = { filename ? filename : "", NULL, -offset };

    unsigned char *buf = NULL;

    int probe_size;



    if (!max_probe_size) {

        max_probe_size = PROBE_BUF_MAX;

    } else if (max_probe_size > PROBE_BUF_MAX) {

        max_probe_size = PROBE_BUF_MAX;

    } else if (max_probe_size < PROBE_BUF_MIN) {

        return AVERROR(EINVAL);

    }



    if (offset >= max_probe_size) {

        return AVERROR(EINVAL);

    }



    for(probe_size= PROBE_BUF_MIN; probe_size<=max_probe_size && !*fmt; probe_size<<=1){

        int ret, score = probe_size < max_probe_size ? AVPROBE_SCORE_MAX/4 : 0;

        int buf_offset = (probe_size == PROBE_BUF_MIN) ? 0 : probe_size>>1;



        if (probe_size < offset) {

            continue;

        }



        /* read probe data */

        buf = av_realloc(buf, probe_size + AVPROBE_PADDING_SIZE);

        if ((ret = get_buffer(*pb, buf + buf_offset, probe_size - buf_offset)) < 0) {

            av_free(buf);

            return ret;

        }

        pd.buf_size += ret;

        pd.buf = &buf[offset];



        memset(pd.buf + pd.buf_size, 0, AVPROBE_PADDING_SIZE);



        /* guess file format */

        *fmt = av_probe_input_format2(&pd, 1, &score);

        if(*fmt){

            if(score <= AVPROBE_SCORE_MAX/4){ //this can only be true in the last iteration

                av_log(logctx, AV_LOG_WARNING, "Format detected only with low score of %d, misdetection possible!\n", score);

            }else

                av_log(logctx, AV_LOG_DEBUG, "Probed with size=%d and score=%d\n", probe_size, score);

        }

    }



    av_free(buf);

    if (url_fseek(*pb, 0, SEEK_SET) < 0) {

        url_fclose(*pb);

        if (url_fopen(pb, filename, URL_RDONLY) < 0)

            return AVERROR(EIO);

    }



    return 0;

}
