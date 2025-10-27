av_cold void ff_dsputil_init_x86(DSPContext *c, AVCodecContext *avctx)

{

    int cpu_flags = av_get_cpu_flags();



#if HAVE_7REGS && HAVE_INLINE_ASM

    if (HAVE_MMX && cpu_flags & AV_CPU_FLAG_CMOV)

        c->add_hfyu_median_prediction = ff_add_hfyu_median_prediction_cmov;

#endif



    if (X86_MMX(cpu_flags)) {

#if HAVE_INLINE_ASM

        const int idct_algo = avctx->idct_algo;



        if (avctx->lowres == 0 && avctx->bits_per_raw_sample <= 8) {

            if (idct_algo == FF_IDCT_AUTO || idct_algo == FF_IDCT_SIMPLEMMX) {

                c->idct_put              = ff_simple_idct_put_mmx;

                c->idct_add              = ff_simple_idct_add_mmx;

                c->idct                  = ff_simple_idct_mmx;

                c->idct_permutation_type = FF_SIMPLE_IDCT_PERM;

            } else if (idct_algo == FF_IDCT_XVIDMMX) {

                if (cpu_flags & AV_CPU_FLAG_SSE2) {

                    c->idct_put              = ff_idct_xvid_sse2_put;

                    c->idct_add              = ff_idct_xvid_sse2_add;

                    c->idct                  = ff_idct_xvid_sse2;

                    c->idct_permutation_type = FF_SSE2_IDCT_PERM;

                } else if (cpu_flags & AV_CPU_FLAG_MMXEXT) {

                    c->idct_put              = ff_idct_xvid_mmxext_put;

                    c->idct_add              = ff_idct_xvid_mmxext_add;

                    c->idct                  = ff_idct_xvid_mmxext;

                } else {

                    c->idct_put              = ff_idct_xvid_mmx_put;

                    c->idct_add              = ff_idct_xvid_mmx_add;

                    c->idct                  = ff_idct_xvid_mmx;

                }

            }

        }

#endif /* HAVE_INLINE_ASM */



        dsputil_init_mmx(c, avctx, cpu_flags);

    }



    if (X86_MMXEXT(cpu_flags))

        dsputil_init_mmxext(c, avctx, cpu_flags);



    if (X86_SSE(cpu_flags))

        dsputil_init_sse(c, avctx, cpu_flags);



    if (X86_SSE2(cpu_flags))

        dsputil_init_sse2(c, avctx, cpu_flags);



    if (EXTERNAL_SSSE3(cpu_flags))

        dsputil_init_ssse3(c, avctx, cpu_flags);



    if (EXTERNAL_SSE4(cpu_flags))

        dsputil_init_sse4(c, avctx, cpu_flags);



    if (CONFIG_ENCODERS)

        ff_dsputilenc_init_mmx(c, avctx);

}
