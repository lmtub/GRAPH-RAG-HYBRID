void av_close_input_file(AVFormatContext *s)

{

    int i;



    if (s->iformat->read_close)

        s->iformat->read_close(s);

    for(i=0;i<s->nb_streams;i++) {

        av_free(s->streams[i]);

    }

    if (s->packet_buffer) {

        AVPacketList *p, *p1;

        p = s->packet_buffer;

        while (p != NULL) {

            p1 = p->next;

            av_free_packet(&p->pkt);

            av_free(p);

            p = p1;

        }

        s->packet_buffer = NULL;

    }

    if (!(s->iformat->flags & AVFMT_NOFILE)) {

        url_fclose(&s->pb);

    }

    av_free(s->priv_data);

    av_free(s);

}
