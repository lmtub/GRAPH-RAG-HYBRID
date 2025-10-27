av_cold void ff_ac3dsp_init_x86(AC3DSPContext *c, int bit_exact)

{

    int mm_flags = av_get_cpu_flags();



    if (EXTERNAL_MMX(mm_flags)) {

        c->ac3_exponent_min = ff_ac3_exponent_min_mmx;

        c->ac3_max_msb_abs_int16 = ff_ac3_max_msb_abs_int16_mmx;

        c->ac3_lshift_int16 = ff_ac3_lshift_int16_mmx;

        c->ac3_rshift_int32 = ff_ac3_rshift_int32_mmx;

    }

    if (EXTERNAL_AMD3DNOW(mm_flags)) {

        c->extract_exponents = ff_ac3_extract_exponents_3dnow;

        if (!bit_exact) {

            c->float_to_fixed24 = ff_float_to_fixed24_3dnow;

        }

    }

    if (EXTERNAL_MMXEXT(mm_flags)) {

        c->ac3_exponent_min = ff_ac3_exponent_min_mmxext;

        c->ac3_max_msb_abs_int16 = ff_ac3_max_msb_abs_int16_mmxext;

    }

    if (EXTERNAL_SSE(mm_flags)) {

        c->float_to_fixed24 = ff_float_to_fixed24_sse;

    }

    if (EXTERNAL_SSE2(mm_flags)) {

        c->ac3_exponent_min = ff_ac3_exponent_min_sse2;

        c->ac3_max_msb_abs_int16 = ff_ac3_max_msb_abs_int16_sse2;

        c->float_to_fixed24 = ff_float_to_fixed24_sse2;

        c->compute_mantissa_size = ff_ac3_compute_mantissa_size_sse2;

        c->extract_exponents = ff_ac3_extract_exponents_sse2;

        if (!(mm_flags & AV_CPU_FLAG_SSE2SLOW)) {

            c->ac3_lshift_int16 = ff_ac3_lshift_int16_sse2;

            c->ac3_rshift_int32 = ff_ac3_rshift_int32_sse2;

        }

    }

    if (EXTERNAL_SSSE3(mm_flags)) {

        c->ac3_max_msb_abs_int16 = ff_ac3_max_msb_abs_int16_ssse3;

        if (!(mm_flags & AV_CPU_FLAG_ATOM)) {

            c->extract_exponents = ff_ac3_extract_exponents_ssse3;

        }

    }



#if HAVE_SSE_INLINE && HAVE_7REGS

    if (INLINE_SSE(mm_flags)) {

        c->downmix = ac3_downmix_sse;

    }

#endif

}
