av_cold void ff_idctdsp_init(IDCTDSPContext *c, AVCodecContext *avctx)

{

    const unsigned high_bit_depth = avctx->bits_per_raw_sample > 8;



    if (avctx->lowres==1) {

        c->idct_put  = ff_jref_idct4_put;

        c->idct_add  = ff_jref_idct4_add;

        c->idct      = ff_j_rev_dct4;

        c->perm_type = FF_IDCT_PERM_NONE;

    } else if (avctx->lowres==2) {

        c->idct_put  = ff_jref_idct2_put;

        c->idct_add  = ff_jref_idct2_add;

        c->idct      = ff_j_rev_dct2;

        c->perm_type = FF_IDCT_PERM_NONE;

    } else if (avctx->lowres==3) {

        c->idct_put  = ff_jref_idct1_put;

        c->idct_add  = ff_jref_idct1_add;

        c->idct      = ff_j_rev_dct1;

        c->perm_type = FF_IDCT_PERM_NONE;

    } else {

        if (avctx->bits_per_raw_sample == 10 || avctx->bits_per_raw_sample == 9) {

            c->idct_put              = ff_simple_idct_put_10;

            c->idct_add              = ff_simple_idct_add_10;

            c->idct                  = ff_simple_idct_10;

            c->perm_type             = FF_IDCT_PERM_NONE;

        } else if (avctx->bits_per_raw_sample == 12) {

            c->idct_put              = ff_simple_idct_put_12;

            c->idct_add              = ff_simple_idct_add_12;

            c->idct                  = ff_simple_idct_12;

            c->perm_type             = FF_IDCT_PERM_NONE;

        } else {

            if (avctx->idct_algo == FF_IDCT_INT) {

                c->idct_put  = ff_jref_idct_put;

                c->idct_add  = ff_jref_idct_add;

                c->idct      = ff_j_rev_dct;

                c->perm_type = FF_IDCT_PERM_LIBMPEG2;

#if CONFIG_FAANIDCT

            } else if (avctx->idct_algo == FF_IDCT_FAAN) {

                c->idct_put  = ff_faanidct_put;

                c->idct_add  = ff_faanidct_add;

                c->idct      = ff_faanidct;

                c->perm_type = FF_IDCT_PERM_NONE;

#endif /* CONFIG_FAANIDCT */

            } else { // accurate/default

                c->idct_put  = ff_simple_idct_put_8;

                c->idct_add  = ff_simple_idct_add_8;

                c->idct      = ff_simple_idct_8;

                c->perm_type = FF_IDCT_PERM_NONE;

            }

        }

    }



    c->put_pixels_clamped        = put_pixels_clamped_c;

    c->put_signed_pixels_clamped = put_signed_pixels_clamped_c;

    c->add_pixels_clamped        = add_pixels_clamped_c;



    if (CONFIG_MPEG4_DECODER && avctx->idct_algo == FF_IDCT_XVID)

        ff_xvid_idct_init(c, avctx);



    if (ARCH_AARCH64)

        ff_idctdsp_init_aarch64(c, avctx, high_bit_depth);

    if (ARCH_ALPHA)

        ff_idctdsp_init_alpha(c, avctx, high_bit_depth);

    if (ARCH_ARM)

        ff_idctdsp_init_arm(c, avctx, high_bit_depth);

    if (ARCH_PPC)

        ff_idctdsp_init_ppc(c, avctx, high_bit_depth);

    if (ARCH_X86)

        ff_idctdsp_init_x86(c, avctx, high_bit_depth);

    if (ARCH_MIPS)

        ff_idctdsp_init_mips(c, avctx, high_bit_depth);



    ff_put_pixels_clamped = c->put_pixels_clamped;

    ff_add_pixels_clamped = c->add_pixels_clamped;



    ff_init_scantable_permutation(c->idct_permutation,

                                  c->perm_type);

}
