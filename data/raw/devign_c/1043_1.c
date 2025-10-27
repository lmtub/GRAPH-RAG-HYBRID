int av_write_trailer(AVFormatContext *s)

{

    int ret, i;



    for (;; ) {

        AVPacket pkt;

        ret = interleave_packet(s, &pkt, NULL, 1);

        if (ret < 0) //FIXME cleanup needed for ret<0 ?

            goto fail;

        if (!ret)

            break;



        ret = s->oformat->write_packet(s, &pkt);

        if (ret >= 0)

            s->streams[pkt.stream_index]->nb_frames++;



        av_free_packet(&pkt);



        if (ret < 0)

            goto fail;

    }



    if (s->oformat->write_trailer)

        ret = s->oformat->write_trailer(s);



    if (!(s->oformat->flags & AVFMT_NOFILE))

        avio_flush(s->pb);



fail:

    for (i = 0; i < s->nb_streams; i++) {

        av_freep(&s->streams[i]->priv_data);

        av_freep(&s->streams[i]->index_entries);

    }

    if (s->oformat->priv_class)

        av_opt_free(s->priv_data);

    av_freep(&s->priv_data);

    return ret;

}
