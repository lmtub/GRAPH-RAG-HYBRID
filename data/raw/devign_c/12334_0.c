DECL_IMDCT_BLOCKS(sse,sse)

#endif

DECL_IMDCT_BLOCKS(sse2,sse)

DECL_IMDCT_BLOCKS(sse3,sse)

DECL_IMDCT_BLOCKS(ssse3,sse)

#endif

#if HAVE_AVX_EXTERNAL

DECL_IMDCT_BLOCKS(avx,avx)

#endif

#endif /* HAVE_YASM */



av_cold void ff_mpadsp_init_x86(MPADSPContext *s)

{

    int cpu_flags = av_get_cpu_flags();



    int i, j;

    for (j = 0; j < 4; j++) {

        for (i = 0; i < 40; i ++) {

            mdct_win_sse[0][j][4*i    ] = ff_mdct_win_float[j    ][i];

            mdct_win_sse[0][j][4*i + 1] = ff_mdct_win_float[j + 4][i];

            mdct_win_sse[0][j][4*i + 2] = ff_mdct_win_float[j    ][i];

            mdct_win_sse[0][j][4*i + 3] = ff_mdct_win_float[j + 4][i];

            mdct_win_sse[1][j][4*i    ] = ff_mdct_win_float[0    ][i];

            mdct_win_sse[1][j][4*i + 1] = ff_mdct_win_float[4    ][i];

            mdct_win_sse[1][j][4*i + 2] = ff_mdct_win_float[j    ][i];

            mdct_win_sse[1][j][4*i + 3] = ff_mdct_win_float[j + 4][i];

        }

    }



#if HAVE_6REGS && HAVE_SSE_INLINE

    if (INLINE_SSE(cpu_flags)) {

        s->apply_window_float = apply_window_mp3;

    }

#endif /* HAVE_SSE_INLINE */



#if HAVE_YASM

#if HAVE_SSE

#if ARCH_X86_32

    if (EXTERNAL_SSE(cpu_flags)) {

        s->imdct36_blocks_float = imdct36_blocks_sse;

    }

#endif

    if (EXTERNAL_SSE2(cpu_flags)) {

        s->imdct36_blocks_float = imdct36_blocks_sse2;

    }

    if (EXTERNAL_SSE3(cpu_flags)) {

        s->imdct36_blocks_float = imdct36_blocks_sse3;

    }

    if (EXTERNAL_SSSE3(cpu_flags)) {

        s->imdct36_blocks_float = imdct36_blocks_ssse3;

    }

#endif

#if HAVE_AVX_EXTERNAL

    if (EXTERNAL_AVX(cpu_flags)) {

        s->imdct36_blocks_float = imdct36_blocks_avx;

    }

#endif

#endif /* HAVE_YASM */

}
