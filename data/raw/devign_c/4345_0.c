void ff_dsputil_init_arm(DSPContext* c, AVCodecContext *avctx)

{

    const int high_bit_depth = avctx->bits_per_raw_sample > 8;

    int cpu_flags = av_get_cpu_flags();



    ff_put_pixels_clamped = c->put_pixels_clamped;

    ff_add_pixels_clamped = c->add_pixels_clamped;



    if (avctx->bits_per_raw_sample <= 8) {

        if(avctx->idct_algo == FF_IDCT_AUTO ||

           avctx->idct_algo == FF_IDCT_ARM){

            c->idct_put              = j_rev_dct_arm_put;

            c->idct_add              = j_rev_dct_arm_add;

            c->idct                  = ff_j_rev_dct_arm;

            c->idct_permutation_type = FF_LIBMPEG2_IDCT_PERM;

        } else if (avctx->idct_algo == FF_IDCT_SIMPLEARM){

            c->idct_put              = simple_idct_arm_put;

            c->idct_add              = simple_idct_arm_add;

            c->idct                  = ff_simple_idct_arm;

            c->idct_permutation_type = FF_NO_IDCT_PERM;

        }

    }



    c->add_pixels_clamped = ff_add_pixels_clamped_arm;



    if (!high_bit_depth) {

    c->put_pixels_tab[0][0] = ff_put_pixels16_arm;

    c->put_pixels_tab[0][1] = ff_put_pixels16_x2_arm;

    c->put_pixels_tab[0][2] = ff_put_pixels16_y2_arm;

    c->put_pixels_tab[0][3] = ff_put_pixels16_xy2_arm;

    c->put_pixels_tab[1][0] = ff_put_pixels8_arm;

    c->put_pixels_tab[1][1] = ff_put_pixels8_x2_arm;

    c->put_pixels_tab[1][2] = ff_put_pixels8_y2_arm;

    c->put_pixels_tab[1][3] = ff_put_pixels8_xy2_arm;



    c->put_no_rnd_pixels_tab[0][0] = ff_put_pixels16_arm;

    c->put_no_rnd_pixels_tab[0][1] = ff_put_no_rnd_pixels16_x2_arm;

    c->put_no_rnd_pixels_tab[0][2] = ff_put_no_rnd_pixels16_y2_arm;

    c->put_no_rnd_pixels_tab[0][3] = ff_put_no_rnd_pixels16_xy2_arm;

    c->put_no_rnd_pixels_tab[1][0] = ff_put_pixels8_arm;

    c->put_no_rnd_pixels_tab[1][1] = ff_put_no_rnd_pixels8_x2_arm;

    c->put_no_rnd_pixels_tab[1][2] = ff_put_no_rnd_pixels8_y2_arm;

    c->put_no_rnd_pixels_tab[1][3] = ff_put_no_rnd_pixels8_xy2_arm;

    }



    if (have_armv5te(cpu_flags)) ff_dsputil_init_armv5te(c, avctx);

    if (have_armv6(cpu_flags))   ff_dsputil_init_armv6(c, avctx);

    if (have_vfp(cpu_flags))     ff_dsputil_init_vfp(c, avctx);

    if (have_neon(cpu_flags))    ff_dsputil_init_neon(c, avctx);

}
