static void free_stream(AVStream **pst)

{

    AVStream *st = *pst;

    int i;



    if (!st)

        return;



    for (i = 0; i < st->nb_side_data; i++)

        av_freep(&st->side_data[i].data);

    av_freep(&st->side_data);



    if (st->parser)

        av_parser_close(st->parser);



    if (st->attached_pic.data)

        av_packet_unref(&st->attached_pic);



    if (st->internal) {

        avcodec_free_context(&st->internal->avctx);

        for (i = 0; i < st->internal->nb_bsfcs; i++) {

            av_bsf_free(&st->internal->bsfcs[i]);

            av_freep(&st->internal->bsfcs);

        }

        av_bsf_free(&st->internal->extract_extradata.bsf);

        av_packet_free(&st->internal->extract_extradata.pkt);

    }

    av_freep(&st->internal);



    av_dict_free(&st->metadata);

    avcodec_parameters_free(&st->codecpar);

    av_freep(&st->probe_data.buf);

    av_freep(&st->index_entries);

#if FF_API_LAVF_AVCTX

FF_DISABLE_DEPRECATION_WARNINGS

    av_freep(&st->codec->extradata);

    av_freep(&st->codec->subtitle_header);

    av_freep(&st->codec);

FF_ENABLE_DEPRECATION_WARNINGS

#endif

    av_freep(&st->priv_data);

    if (st->info)

        av_freep(&st->info->duration_error);

    av_freep(&st->info);

    av_freep(&st->recommended_encoder_configuration);

    av_freep(&st->priv_pts);



    av_freep(pst);

}
