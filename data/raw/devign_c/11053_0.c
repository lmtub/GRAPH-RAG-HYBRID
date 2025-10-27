AVCodecParserContext *av_parser_init(int codec_id)

{

    AVCodecParserContext *s;

    AVCodecParser *parser;

    int ret;



    if(codec_id == CODEC_ID_NONE)

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

        return NULL;

    s->parser = parser;

    if (parser->priv_data_size) {

        s->priv_data = av_mallocz(parser->priv_data_size);

        if (!s->priv_data) {

            av_free(s);

            return NULL;

        }

    }

    if (parser->parser_init) {

        ret = parser->parser_init(s);

        if (ret != 0) {

            av_free(s->priv_data);

            av_free(s);

            return NULL;

        }

    }

    s->fetch_timestamp=1;

    s->pict_type = FF_I_TYPE;

    s->key_frame = -1;

    s->convergence_duration = 0;

    s->dts_sync_point       = INT_MIN;

    s->dts_ref_dts_delta    = INT_MIN;

    s->pts_dts_delta        = INT_MIN;

    return s;

}
