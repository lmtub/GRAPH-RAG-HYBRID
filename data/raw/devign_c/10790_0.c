static int mpegps_read_header(AVFormatContext *s,

                              AVFormatParameters *ap)

{

    MpegDemuxContext *m = s->priv_data;

    uint8_t buffer[8192];

    char *p;



    m->header_state = 0xff;

    s->ctx_flags |= AVFMTCTX_NOHEADER;



    get_buffer(&s->pb, buffer, sizeof(buffer));

    if ((p=memchr(buffer, 'S', sizeof(buffer))))

        if (!memcmp(p, "Sofdec", 6))

            m->sofdec = 1;

    url_fseek(&s->pb, -sizeof(buffer), SEEK_CUR);



    /* no need to do more */

    return 0;

}
