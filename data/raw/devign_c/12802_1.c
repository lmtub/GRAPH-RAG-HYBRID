av_cold int ff_ac3_encode_close(AVCodecContext *avctx)

{

    int blk, ch;

    AC3EncodeContext *s = avctx->priv_data;



    av_freep(&s->windowed_samples);


    for (ch = 0; ch < s->channels; ch++)

        av_freep(&s->planar_samples[ch]);

    av_freep(&s->planar_samples);

    av_freep(&s->bap_buffer);

    av_freep(&s->bap1_buffer);

    av_freep(&s->mdct_coef_buffer);

    av_freep(&s->fixed_coef_buffer);

    av_freep(&s->exp_buffer);

    av_freep(&s->grouped_exp_buffer);

    av_freep(&s->psd_buffer);

    av_freep(&s->band_psd_buffer);

    av_freep(&s->mask_buffer);

    av_freep(&s->qmant_buffer);

    av_freep(&s->cpl_coord_exp_buffer);

    av_freep(&s->cpl_coord_mant_buffer);

    for (blk = 0; blk < s->num_blocks; blk++) {

        AC3Block *block = &s->blocks[blk];

        av_freep(&block->mdct_coef);

        av_freep(&block->fixed_coef);

        av_freep(&block->exp);

        av_freep(&block->grouped_exp);

        av_freep(&block->psd);

        av_freep(&block->band_psd);

        av_freep(&block->mask);

        av_freep(&block->qmant);

        av_freep(&block->cpl_coord_exp);

        av_freep(&block->cpl_coord_mant);

    }



    s->mdct_end(s);



    return 0;

}