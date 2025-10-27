static int decode_end(AVCodecContext *avctx)

{

    SmackVContext * const smk = (SmackVContext *)avctx->priv_data;



    if(smk->mmap_tbl)

        av_free(smk->mmap_tbl);

    if(smk->mclr_tbl)

        av_free(smk->mclr_tbl);

    if(smk->full_tbl)

        av_free(smk->full_tbl);

    if(smk->type_tbl)

        av_free(smk->type_tbl);



    if (smk->pic.data[0])

        avctx->release_buffer(avctx, &smk->pic);



    return 0;

}
