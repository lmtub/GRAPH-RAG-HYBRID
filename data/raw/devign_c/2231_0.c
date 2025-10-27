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

        }

    }



    /* remove references, if any (XXX: do it faster) */

    for(c1 = first_http_ctx; c1 != NULL; c1 = c1->next) {

        if (c1->rtsp_c == c)

            c1->rtsp_c = NULL;

    }



    /* remove connection associated resources */

    if (c->fd >= 0)

        close(c->fd);

    if (c->fmt_in) {

        /* close each frame parser */

        for(i=0;i<c->fmt_in->nb_streams;i++) {

            st = c->fmt_in->streams[i];

            if (st->codec->codec) {

                avcodec_close(st->codec);

            }

        }

        av_close_input_file(c->fmt_in);

    }



    /* free RTP output streams if any */

    nb_streams = 0;

    if (c->stream) 

        nb_streams = c->stream->nb_streams;

    

    for(i=0;i<nb_streams;i++) {

        ctx = c->rtp_ctx[i];

        if (ctx) {

            av_write_trailer(ctx);

            av_free(ctx);

        }

        h = c->rtp_handles[i];

        if (h) {

            url_close(h);

        }

    }

    

    ctx = &c->fmt_ctx;



    if (!c->last_packet_sent) {

        if (ctx->oformat) {

            /* prepare header */

            if (url_open_dyn_buf(&ctx->pb) >= 0) {

                av_write_trailer(ctx);

                url_close_dyn_buf(&ctx->pb, &c->pb_buffer);

            }

        }

    }



    for(i=0; i<ctx->nb_streams; i++) 

        av_free(ctx->streams[i]) ; 



    if (c->stream)

        current_bandwidth -= c->stream->bandwidth;

    av_freep(&c->pb_buffer);

    av_freep(&c->packet_buffer);

    av_free(c->buffer);

    av_free(c);

    nb_connections--;

}
