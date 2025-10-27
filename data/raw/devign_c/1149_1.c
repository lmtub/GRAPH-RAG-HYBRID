AVCodecParserContext *av_parser_init(int codec_id)

{

    AVCodecParserContext *s = NULL;

    AVCodecParser *parser;

    int ret;



    if(codec_id == AV_CODEC_ID_NONE)

        return NULL;



    for(parser = av_first_parser; parser != NULL; parser = parser->next) {

        if (parser->codec_ids[0] == codec_id ||

            parser->codec_ids[1] == codec_id ||

            parser->codec_ids[2] == codec_id ||

            parser->codec_ids[3] == codec_id ||

            parser->codec_ids[4] == codec_id)

            goto found;

    }

    return NULL;

 found:

    s = av_mallocz(sizeof(AVCodecParserContext));

    if (!s)

        goto err_out;

    s->parser = parser;

    s->priv_data = av_mallocz(parser->priv_data_size);

    if (!s->priv_data)

        goto err_out;

    s->fetch_timestamp=1;

    s->pict_type = AV_PICTURE_TYPE_I;

    if (parser->parser_init) {

        if (ff_lock_avcodec(NULL) < 0)

            goto err_out;

        ret = parser->parser_init(s);

        ff_unlock_avcodec();

        if (ret != 0)

            goto err_out;

    }

    s->key_frame = -1;

    s->convergence_duration = 0;

    s->dts_sync_point       = INT_MIN;

    s->dts_ref_dts_delta    = INT_MIN;

    s->pts_dts_delta        = INT_MIN;

    return s;



err_out:

    if (s)

        av_freep(&s->priv_data);

    av_free(s);

    return NULL;

}
