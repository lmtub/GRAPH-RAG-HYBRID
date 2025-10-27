static av_cold int dnxhd_encode_end(AVCodecContext *avctx)

{

    DNXHDEncContext *ctx = avctx->priv_data;

    int max_level        = 1 << (ctx->cid_table->bit_depth + 2);

    int i;



    av_free(ctx->vlc_codes - max_level * 2);

    av_free(ctx->vlc_bits - max_level * 2);

    av_freep(&ctx->run_codes);

    av_freep(&ctx->run_bits);



    av_freep(&ctx->mb_bits);

    av_freep(&ctx->mb_qscale);

    av_freep(&ctx->mb_rc);

    av_freep(&ctx->mb_cmp);

    av_freep(&ctx->slice_size);

    av_freep(&ctx->slice_offs);



    av_freep(&ctx->qmatrix_c);

    av_freep(&ctx->qmatrix_l);

    av_freep(&ctx->qmatrix_c16);

    av_freep(&ctx->qmatrix_l16);



    for (i = 1; i < avctx->thread_count; i++)

        av_freep(&ctx->thread[i]);



    av_frame_free(&avctx->coded_frame);



    return 0;

}
