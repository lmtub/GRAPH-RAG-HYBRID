int ff_wma_end(AVCodecContext *avctx)

{

    WMACodecContext *s = avctx->priv_data;

    int i;



    for(i = 0; i < s->nb_block_sizes; i++)

        ff_mdct_end(&s->mdct_ctx[i]);

    for(i = 0; i < s->nb_block_sizes; i++)

        av_free(s->windows[i]);



    if (s->use_exp_vlc) {

        free_vlc(&s->exp_vlc);

    }

    if (s->use_noise_coding) {

        free_vlc(&s->hgain_vlc);

    }

    for(i = 0;i < 2; i++) {

        free_vlc(&s->coef_vlc[i]);

        av_free(s->run_table[i]);

        av_free(s->level_table[i]);


    }



    return 0;

}