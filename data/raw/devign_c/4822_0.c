static int ftp_close(URLContext *h)

{

    FTPContext *s = h->priv_data;



    av_dlog(h, "ftp protocol close\n");



    ftp_close_both_connections(s);

    av_freep(&s->user);

    av_freep(&s->password);

    av_freep(&s->hostname);

    av_freep(&s->path);

    av_freep(&s->features);



    return 0;

}
