static void hevc_parser_close(AVCodecParserContext *s)

{

    HEVCParserContext *ctx = s->priv_data;

    int i;



#if ADVANCED_PARSER

    HEVCContext  *h  = &ctx->h;



    for (i = 0; i < FF_ARRAY_ELEMS(h->ps.vps_list); i++)

        av_buffer_unref(&h->ps.vps_list[i]);

    for (i = 0; i < FF_ARRAY_ELEMS(h->ps.sps_list); i++)

        av_buffer_unref(&h->ps.sps_list[i]);

    for (i = 0; i < FF_ARRAY_ELEMS(h->ps.pps_list); i++)

        av_buffer_unref(&h->ps.pps_list[i]);



    h->ps.sps = NULL;



    av_freep(&h->HEVClc);

#endif



    for (i = 0; i < FF_ARRAY_ELEMS(ctx->ps.vps_list); i++)

        av_buffer_unref(&ctx->ps.vps_list[i]);

    for (i = 0; i < FF_ARRAY_ELEMS(ctx->ps.sps_list); i++)

        av_buffer_unref(&ctx->ps.sps_list[i]);

    for (i = 0; i < FF_ARRAY_ELEMS(ctx->ps.pps_list); i++)

        av_buffer_unref(&ctx->ps.pps_list[i]);



    ctx->ps.sps = NULL;



    ff_h2645_packet_uninit(&ctx->pkt);



    av_freep(&ctx->pc.buffer);

}
