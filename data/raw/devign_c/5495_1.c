static int tee_write_trailer(AVFormatContext *avf)

{

    TeeContext *tee = avf->priv_data;

    AVFormatContext *avf2;

    int ret_all = 0, ret;

    unsigned i;



    for (i = 0; i < tee->nb_slaves; i++) {

        avf2 = tee->slaves[i].avf;

        if ((ret = av_write_trailer(avf2)) < 0)

            if (!ret_all)

                ret_all = ret;

        if (!(avf2->oformat->flags & AVFMT_NOFILE))

            ff_format_io_close(avf2, &avf2->pb);

    }

    close_slaves(avf);

    return ret_all;

}
