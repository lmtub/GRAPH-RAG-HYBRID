static int hls_start(AVFormatContext *s)

{

    HLSContext *c = s->priv_data;

    AVFormatContext *oc = c->avf;

    int err = 0;



    if (c->wrap)

        c->number %= c->wrap;



    if (av_get_frame_filename(oc->filename, sizeof(oc->filename),

                              c->basename, c->number++) < 0)

        return AVERROR(EINVAL);



    if ((err = avio_open2(&oc->pb, oc->filename, AVIO_FLAG_WRITE,

                          &s->interrupt_callback, NULL)) < 0)

        return err;



    if (oc->oformat->priv_class && oc->priv_data)

        av_opt_set(oc->priv_data, "mpegts_flags", "resend_headers", 0);



    return 0;

}
