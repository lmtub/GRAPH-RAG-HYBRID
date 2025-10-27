av_cold void ff_MPV_common_init_x86(MpegEncContext *s)

{

#if HAVE_INLINE_ASM

    int cpu_flags = av_get_cpu_flags();



    if (cpu_flags & AV_CPU_FLAG_MMX) {

        s->dct_unquantize_h263_intra = dct_unquantize_h263_intra_mmx;

        s->dct_unquantize_h263_inter = dct_unquantize_h263_inter_mmx;

        s->dct_unquantize_mpeg1_intra = dct_unquantize_mpeg1_intra_mmx;

        s->dct_unquantize_mpeg1_inter = dct_unquantize_mpeg1_inter_mmx;

        if(!(s->flags & CODEC_FLAG_BITEXACT))

            s->dct_unquantize_mpeg2_intra = dct_unquantize_mpeg2_intra_mmx;

        s->dct_unquantize_mpeg2_inter = dct_unquantize_mpeg2_inter_mmx;



        if (cpu_flags & AV_CPU_FLAG_SSE2) {

            s->denoise_dct= denoise_dct_sse2;

        } else {

                s->denoise_dct= denoise_dct_mmx;

        }

    }

#endif /* HAVE_INLINE_ASM */

}
