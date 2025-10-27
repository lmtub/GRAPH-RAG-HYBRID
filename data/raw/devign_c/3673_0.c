rdt_free_context (PayloadContext *rdt)

{

    int i;



    for (i = 0; i < MAX_STREAMS; i++)

        if (rdt->rmst[i]) {

            ff_rm_free_rmstream(rdt->rmst[i]);

            av_freep(&rdt->rmst[i]);

        }

    if (rdt->rmctx)

        av_close_input_stream(rdt->rmctx);

    av_freep(&rdt->mlti_data);

    av_free(rdt);

}
