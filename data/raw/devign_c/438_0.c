av_cold void ff_dsputil_init_armv6(DSPContext *c, AVCodecContext *avctx)

{

    const int high_bit_depth = avctx->bits_per_raw_sample > 8;



    if (avctx->bits_per_raw_sample <= 8 &&

        (avctx->idct_algo == FF_IDCT_AUTO ||

         avctx->idct_algo == FF_IDCT_SIMPLEARMV6)) {

        c->idct_put              = ff_simple_idct_put_armv6;

        c->idct_add              = ff_simple_idct_add_armv6;

        c->idct                  = ff_simple_idct_armv6;

        c->idct_permutation_type = FF_LIBMPEG2_IDCT_PERM;

    }



    if (!high_bit_depth) {

    c->put_pixels_tab[0][0] = ff_put_pixels16_armv6;

    c->put_pixels_tab[0][1] = ff_put_pixels16_x2_armv6;

    c->put_pixels_tab[0][2] = ff_put_pixels16_y2_armv6;

/*     c->put_pixels_tab[0][3] = ff_put_pixels16_xy2_armv6; */

    c->put_pixels_tab[1][0] = ff_put_pixels8_armv6;

    c->put_pixels_tab[1][1] = ff_put_pixels8_x2_armv6;

    c->put_pixels_tab[1][2] = ff_put_pixels8_y2_armv6;

/*     c->put_pixels_tab[1][3] = ff_put_pixels8_xy2_armv6; */



    c->put_no_rnd_pixels_tab[0][0] = ff_put_pixels16_armv6;

    c->put_no_rnd_pixels_tab[0][1] = ff_put_pixels16_x2_no_rnd_armv6;

    c->put_no_rnd_pixels_tab[0][2] = ff_put_pixels16_y2_no_rnd_armv6;

/*     c->put_no_rnd_pixels_tab[0][3] = ff_put_pixels16_xy2_no_rnd_armv6; */

    c->put_no_rnd_pixels_tab[1][0] = ff_put_pixels8_armv6;

    c->put_no_rnd_pixels_tab[1][1] = ff_put_pixels8_x2_no_rnd_armv6;

    c->put_no_rnd_pixels_tab[1][2] = ff_put_pixels8_y2_no_rnd_armv6;

/*     c->put_no_rnd_pixels_tab[1][3] = ff_put_pixels8_xy2_no_rnd_armv6; */



    c->avg_pixels_tab[0][0] = ff_avg_pixels16_armv6;

    c->avg_pixels_tab[1][0] = ff_avg_pixels8_armv6;

    }



    if (!high_bit_depth)

        c->get_pixels = ff_get_pixels_armv6;

    c->add_pixels_clamped = ff_add_pixels_clamped_armv6;

    c->diff_pixels = ff_diff_pixels_armv6;



    c->pix_abs[0][0] = ff_pix_abs16_armv6;

    c->pix_abs[0][1] = ff_pix_abs16_x2_armv6;

    c->pix_abs[0][2] = ff_pix_abs16_y2_armv6;



    c->pix_abs[1][0] = ff_pix_abs8_armv6;



    c->sad[0] = ff_pix_abs16_armv6;

    c->sad[1] = ff_pix_abs8_armv6;



    c->sse[0] = ff_sse16_armv6;



    c->pix_norm1 = ff_pix_norm1_armv6;

    c->pix_sum   = ff_pix_sum_armv6;

}
