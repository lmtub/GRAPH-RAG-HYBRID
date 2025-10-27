rdt_new_context (void)

{

    PayloadContext *rdt = av_mallocz(sizeof(PayloadContext));



    avformat_open_input(&rdt->rmctx, "", &ff_rdt_demuxer, NULL);



    return rdt;

}
