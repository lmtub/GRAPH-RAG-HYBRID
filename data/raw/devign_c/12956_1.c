static void writer_close(WriterContext **wctx)

{

    int i;



    if (!*wctx)

        return;



    if ((*wctx)->writer->uninit)

        (*wctx)->writer->uninit(*wctx);

    for (i = 0; i < SECTION_MAX_NB_LEVELS; i++)

        av_bprint_finalize(&(*wctx)->section_pbuf[i], NULL);

    if ((*wctx)->writer->priv_class)

        av_opt_free((*wctx)->priv);

    av_freep(&((*wctx)->priv));


    av_freep(wctx);

}