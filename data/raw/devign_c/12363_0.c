void ff_dsputil_init_mmx(DSPContext *c, AVCodecContext *avctx)

{

    int mm_flags = av_get_cpu_flags();



    if (mm_flags & AV_CPU_FLAG_MMX) {

#if HAVE_INLINE_ASM

        const int idct_algo = avctx->idct_algo;



        if (avctx->bits_per_raw_sample <= 8) {

            if (idct_algo == FF_IDCT_AUTO || idct_algo == FF_IDCT_SIMPLEMMX) {

                c->idct_put              = ff_simple_idct_put_mmx;

                c->idct_add              = ff_simple_idct_add_mmx;

                c->idct                  = ff_simple_idct_mmx;

                c->idct_permutation_type = FF_SIMPLE_IDCT_PERM;

            } else if (idct_algo == FF_IDCT_CAVS) {

                    c->idct_permutation_type = FF_TRANSPOSE_IDCT_PERM;

            } else if (idct_algo == FF_IDCT_XVIDMMX) {

                if (mm_flags & AV_CPU_FLAG_SSE2) {

                    c->idct_put              = ff_idct_xvid_sse2_put;

                    c->idct_add              = ff_idct_xvid_sse2_add;

                    c->idct                  = ff_idct_xvid_sse2;

                    c->idct_permutation_type = FF_SSE2_IDCT_PERM;

                } else if (mm_flags & AV_CPU_FLAG_MMXEXT) {

                    c->idct_put              = ff_idct_xvid_mmx2_put;

                    c->idct_add              = ff_idct_xvid_mmx2_add;

                    c->idct                  = ff_idct_xvid_mmx2;

                } else {

                    c->idct_put              = ff_idct_xvid_mmx_put;

                    c->idct_add              = ff_idct_xvid_mmx_add;

                    c->idct                  = ff_idct_xvid_mmx;

                }

            }

        }

#endif /* HAVE_INLINE_ASM */



        dsputil_init_mmx(c, avctx, mm_flags);

    }



    if (mm_flags & AV_CPU_FLAG_MMXEXT)

        dsputil_init_mmx2(c, avctx, mm_flags);



    if (mm_flags & AV_CPU_FLAG_3DNOW && HAVE_AMD3DNOW)

        dsputil_init_3dnow(c, avctx, mm_flags);



    if (mm_flags & AV_CPU_FLAG_3DNOWEXT && HAVE_AMD3DNOWEXT)

        dsputil_init_3dnowext(c, avctx, mm_flags);



    if (mm_flags & AV_CPU_FLAG_SSE && HAVE_SSE)

        dsputil_init_sse(c, avctx, mm_flags);



    if (mm_flags & AV_CPU_FLAG_SSE2)

        dsputil_init_sse2(c, avctx, mm_flags);



    if (mm_flags & AV_CPU_FLAG_SSSE3)

        dsputil_init_ssse3(c, avctx, mm_flags);



    if (mm_flags & AV_CPU_FLAG_SSE4 && HAVE_SSE)

        dsputil_init_sse4(c, avctx, mm_flags);



    if (mm_flags & AV_CPU_FLAG_AVX)

        dsputil_init_avx(c, avctx, mm_flags);



    if (CONFIG_ENCODERS)

        ff_dsputilenc_init_mmx(c, avctx);

}
