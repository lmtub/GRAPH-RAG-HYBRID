void avformat_free_context(AVFormatContext *s)

{

    int i;

    AVStream *st;



    av_opt_free(s);

    if (s->iformat && s->iformat->priv_class && s->priv_data)

        av_opt_free(s->priv_data);



    for(i=0;i<s->nb_streams;i++) {

        /* free all data in a stream component */

        st = s->streams[i];

        if (st->parser) {

            av_parser_close(st->parser);

            av_free_packet(&st->cur_pkt);

        }

        if (st->attached_pic.data)

            av_free_packet(&st->attached_pic);

        av_dict_free(&st->metadata);

        av_free(st->index_entries);

        av_free(st->codec->extradata);

        av_free(st->codec->subtitle_header);

        av_free(st->codec);

        av_free(st->priv_data);

        av_free(st->info);

        av_free(st);

    }

    for(i=s->nb_programs-1; i>=0; i--) {

        av_dict_free(&s->programs[i]->metadata);

        av_freep(&s->programs[i]->stream_index);

        av_freep(&s->programs[i]);

    }

    av_freep(&s->programs);

    av_freep(&s->priv_data);

    while(s->nb_chapters--) {

        av_dict_free(&s->chapters[s->nb_chapters]->metadata);

        av_free(s->chapters[s->nb_chapters]);

    }

    av_freep(&s->chapters);

    av_dict_free(&s->metadata);

    av_freep(&s->streams);

    av_free(s);

}
