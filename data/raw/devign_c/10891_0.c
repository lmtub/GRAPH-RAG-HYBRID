rdt_new_extradata (void)

{

    PayloadContext *rdt = av_mallocz(sizeof(PayloadContext));



    av_open_input_stream(&rdt->rmctx, NULL, "", &rdt_demuxer, NULL);



    return rdt;

}
