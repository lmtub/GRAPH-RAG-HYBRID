static void asfrtp_close_context(PayloadContext *asf)

{

    ffio_free_dyn_buf(&asf->pktbuf);

    av_freep(&asf->buf);

    av_free(asf);

}
