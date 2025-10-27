static void close_connection(HTTPContext *c)
{
    HTTPContext **cp, *c1;
    int i, nb_streams;
    AVFormatContext *ctx;
    URLContext *h;
    AVStream *st;
    /* remove connection from list */
    cp = &first_http_ctx;
    while ((*cp) != NULL) {
        c1 = *cp;
        if (c1 == c) {
            *cp = c->next;
        } else {
            cp = &c1->next;
    /* remove connection associated resources */
    if (c->fd >= 0)
        close(c->fd);
    if (c->fmt_in) {
        /* close each frame parser */
        for(i=0;i<c->fmt_in->nb_streams;i++) {
            st = c->fmt_in->streams[i];
            if (st->codec.codec) {
                avcodec_close(&st->codec);
        av_close_input_file(c->fmt_in);
    /* free RTP output streams if any */
    nb_streams = 0;
    if (c->stream) 
        nb_streams = c->stream->nb_streams;
    for(i=0;i<nb_streams;i++) {
        ctx = c->rtp_ctx[i];
        if (ctx) {
            av_free(ctx);
        h = c->rtp_handles[i];
        if (h) {
            url_close(h);
    if (c->stream)
        current_bandwidth -= c->stream->bandwidth;
    av_freep(&c->pb_buffer);
    av_free(c->buffer);
    av_free(c);
    nb_connections--;