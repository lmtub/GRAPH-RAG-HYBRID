static int tee_write_header(AVFormatContext *avf)

{

    TeeContext *tee = avf->priv_data;

    unsigned nb_slaves = 0, i;

    const char *filename = avf->filename;

    char *slaves[MAX_SLAVES];

    int ret;



    while (*filename) {

        if (nb_slaves == MAX_SLAVES) {

            av_log(avf, AV_LOG_ERROR, "Maximum %d slave muxers reached.\n",

                   MAX_SLAVES);

            ret = AVERROR_PATCHWELCOME;

            goto fail;

        }

        if (!(slaves[nb_slaves++] = av_get_token(&filename, slave_delim))) {

            ret = AVERROR(ENOMEM);

            goto fail;

        }

        if (strspn(filename, slave_delim))

            filename++;

    }



    for (i = 0; i < nb_slaves; i++) {

        if ((ret = open_slave(avf, slaves[i], &tee->slaves[i])) < 0)

            goto fail;

        log_slave(&tee->slaves[i], avf, AV_LOG_VERBOSE);

        av_freep(&slaves[i]);

    }



    tee->nb_slaves = nb_slaves;



    for (i = 0; i < avf->nb_streams; i++) {

        int j, mapped = 0;

        for (j = 0; j < tee->nb_slaves; j++)

            mapped += tee->slaves[j].stream_map[i] >= 0;

        if (!mapped)

            av_log(avf, AV_LOG_WARNING, "Input stream #%d is not mapped "

                   "to any slave.\n", i);

    }

    return 0;



fail:

    for (i = 0; i < nb_slaves; i++)

        av_freep(&slaves[i]);

    close_slaves(avf);

    return ret;

}
