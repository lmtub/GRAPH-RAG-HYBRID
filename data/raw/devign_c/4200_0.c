av_cold void ff_audio_mix_init_x86(AudioMix *am)

{

#if HAVE_YASM

    int mm_flags = av_get_cpu_flags();



    if (mm_flags & AV_CPU_FLAG_SSE && HAVE_SSE) {

        ff_audio_mix_set_func(am, AV_SAMPLE_FMT_FLTP, AV_MIX_COEFF_TYPE_FLT,

                              2, 1, 16, 8, "SSE", ff_mix_2_to_1_fltp_flt_sse);

        ff_audio_mix_set_func(am, AV_SAMPLE_FMT_FLTP, AV_MIX_COEFF_TYPE_FLT,

                              1, 2, 16, 4, "SSE", ff_mix_1_to_2_fltp_flt_sse);

    }

    if (mm_flags & AV_CPU_FLAG_SSE2 && HAVE_SSE) {

        ff_audio_mix_set_func(am, AV_SAMPLE_FMT_S16P, AV_MIX_COEFF_TYPE_FLT,

                              2, 1, 16, 8, "SSE2", ff_mix_2_to_1_s16p_flt_sse2);

        ff_audio_mix_set_func(am, AV_SAMPLE_FMT_S16P, AV_MIX_COEFF_TYPE_Q8,

                              2, 1, 16, 8, "SSE2", ff_mix_2_to_1_s16p_q8_sse2);

        ff_audio_mix_set_func(am, AV_SAMPLE_FMT_S16P, AV_MIX_COEFF_TYPE_FLT,

                              1, 2, 16, 8, "SSE2", ff_mix_1_to_2_s16p_flt_sse2);

    }

    if (mm_flags & AV_CPU_FLAG_SSE4 && HAVE_SSE) {

        ff_audio_mix_set_func(am, AV_SAMPLE_FMT_S16P, AV_MIX_COEFF_TYPE_FLT,

                              2, 1, 16, 8, "SSE4", ff_mix_2_to_1_s16p_flt_sse4);

        ff_audio_mix_set_func(am, AV_SAMPLE_FMT_S16P, AV_MIX_COEFF_TYPE_FLT,

                              1, 2, 16, 8, "SSE4", ff_mix_1_to_2_s16p_flt_sse4);

    }

    if (mm_flags & AV_CPU_FLAG_AVX && HAVE_AVX) {

        ff_audio_mix_set_func(am, AV_SAMPLE_FMT_FLTP, AV_MIX_COEFF_TYPE_FLT,

                              2, 1, 32, 16, "AVX", ff_mix_2_to_1_fltp_flt_avx);

        ff_audio_mix_set_func(am, AV_SAMPLE_FMT_FLTP, AV_MIX_COEFF_TYPE_FLT,

                              1, 2, 32, 8, "AVX", ff_mix_1_to_2_fltp_flt_avx);

        ff_audio_mix_set_func(am, AV_SAMPLE_FMT_S16P, AV_MIX_COEFF_TYPE_FLT,

                              1, 2, 16, 8, "AVX", ff_mix_1_to_2_s16p_flt_avx);

    }



    SET_MIX_3_8_TO_1_2(3)

    SET_MIX_3_8_TO_1_2(4)

    SET_MIX_3_8_TO_1_2(5)

    SET_MIX_3_8_TO_1_2(6)

    SET_MIX_3_8_TO_1_2(7)

    SET_MIX_3_8_TO_1_2(8)

#endif /* HAVE_YASM */

}
