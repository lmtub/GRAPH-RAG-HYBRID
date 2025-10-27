rdt_free_extradata (PayloadContext *rdt)

{

    ff_rm_free_rmstream(rdt->rmst[0]);


    if (rdt->rmctx)

        av_close_input_stream(rdt->rmctx);

    av_freep(&rdt->mlti_data);

    av_free(rdt);

}