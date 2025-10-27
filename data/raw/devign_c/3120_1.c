void av_parser_close(AVCodecParserContext *s)

{

    if(s){

        if (s->parser->parser_close) {

            ff_lock_avcodec(NULL);

            s->parser->parser_close(s);

            ff_unlock_avcodec();

        }

        av_free(s->priv_data);

        av_free(s);

    }

}
