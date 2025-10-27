av_cold void ff_audio_convert_init_x86(AudioConvert *ac)

{

    int cpu_flags = av_get_cpu_flags();



    if (EXTERNAL_MMX(cpu_flags)) {

        ff_audio_convert_set_func(ac, AV_SAMPLE_FMT_S16, AV_SAMPLE_FMT_S32,

                                  0, 1, 8, "MMX", ff_conv_s32_to_s16_mmx);

        ff_audio_convert_set_func(ac, AV_SAMPLE_FMT_FLT, AV_SAMPLE_FMT_FLTP,

                                  6, 1, 4, "MMX", ff_conv_fltp_to_flt_6ch_mmx);

    }

    if (EXTERNAL_SSE(cpu_flags)) {

        ff_audio_convert_set_func(ac, AV_SAMPLE_FMT_S16, AV_SAMPLE_FMT_FLTP,

                                  6, 1, 2, "SSE", ff_conv_fltp_to_s16_6ch_sse);

        ff_audio_convert_set_func(ac, AV_SAMPLE_FMT_FLT, AV_SAMPLE_FMT_FLTP,

                                  2, 16, 8, "SSE", ff_conv_fltp_to_flt_2ch_sse);

        ff_audio_convert_set_func(ac, AV_SAMPLE_FMT_FLTP, AV_SAMPLE_FMT_FLT,

                                  2, 16, 4, "SSE", ff_conv_flt_to_fltp_2ch_sse);

    }

    if (EXTERNAL_SSE2(cpu_flags)) {

        if (!(cpu_flags & AV_CPU_FLAG_SSE2SLOW)) {

            ff_audio_convert_set_func(ac, AV_SAMPLE_FMT_S16, AV_SAMPLE_FMT_S32,

                                      0, 16, 16, "SSE2", ff_conv_s32_to_s16_sse2);

            ff_audio_convert_set_func(ac, AV_SAMPLE_FMT_S16, AV_SAMPLE_FMT_S16P,

                                      6, 16, 8, "SSE2", ff_conv_s16p_to_s16_6ch_sse2);

            ff_audio_convert_set_func(ac, AV_SAMPLE_FMT_S16, AV_SAMPLE_FMT_FLTP,

                                      6, 16, 4, "SSE2", ff_conv_fltp_to_s16_6ch_sse2);

        } else {

            ff_audio_convert_set_func(ac, AV_SAMPLE_FMT_S16, AV_SAMPLE_FMT_S16P,

                                      6, 1, 4, "SSE2SLOW", ff_conv_s16p_to_s16_6ch_sse2slow);

        }

        ff_audio_convert_set_func(ac, AV_SAMPLE_FMT_S32, AV_SAMPLE_FMT_S16,

                                  0, 16, 8, "SSE2", ff_conv_s16_to_s32_sse2);

        ff_audio_convert_set_func(ac, AV_SAMPLE_FMT_FLT, AV_SAMPLE_FMT_S16,

                                  0, 16, 8, "SSE2", ff_conv_s16_to_flt_sse2);

        ff_audio_convert_set_func(ac, AV_SAMPLE_FMT_FLT, AV_SAMPLE_FMT_S32,

                                  0, 16, 8, "SSE2", ff_conv_s32_to_flt_sse2);

        ff_audio_convert_set_func(ac, AV_SAMPLE_FMT_S16, AV_SAMPLE_FMT_FLT,

                                  0, 16, 16, "SSE2", ff_conv_flt_to_s16_sse2);

        ff_audio_convert_set_func(ac, AV_SAMPLE_FMT_S32, AV_SAMPLE_FMT_FLT,

                                  0, 16, 16, "SSE2", ff_conv_flt_to_s32_sse2);

        ff_audio_convert_set_func(ac, AV_SAMPLE_FMT_S16, AV_SAMPLE_FMT_S16P,

                                  2, 16, 16, "SSE2", ff_conv_s16p_to_s16_2ch_sse2);

        ff_audio_convert_set_func(ac, AV_SAMPLE_FMT_FLT, AV_SAMPLE_FMT_S16P,

                                  2, 16, 8, "SSE2", ff_conv_s16p_to_flt_2ch_sse2);

        ff_audio_convert_set_func(ac, AV_SAMPLE_FMT_FLT, AV_SAMPLE_FMT_S16P,

                                  6, 16, 4, "SSE2", ff_conv_s16p_to_flt_6ch_sse2);

        ff_audio_convert_set_func(ac, AV_SAMPLE_FMT_S16, AV_SAMPLE_FMT_FLTP,

                                  2, 16, 4, "SSE2", ff_conv_fltp_to_s16_2ch_sse2);

        ff_audio_convert_set_func(ac, AV_SAMPLE_FMT_S16P, AV_SAMPLE_FMT_S16,

                                  2, 16, 8, "SSE2", ff_conv_s16_to_s16p_2ch_sse2);

        ff_audio_convert_set_func(ac, AV_SAMPLE_FMT_S16P, AV_SAMPLE_FMT_S16,

                                  6, 16, 4, "SSE2", ff_conv_s16_to_s16p_6ch_sse2);

        ff_audio_convert_set_func(ac, AV_SAMPLE_FMT_FLTP, AV_SAMPLE_FMT_S16,

                                  2, 16, 8, "SSE2", ff_conv_s16_to_fltp_2ch_sse2);

        ff_audio_convert_set_func(ac, AV_SAMPLE_FMT_FLTP, AV_SAMPLE_FMT_S16,

                                  6, 16, 4, "SSE2", ff_conv_s16_to_fltp_6ch_sse2);

        ff_audio_convert_set_func(ac, AV_SAMPLE_FMT_S16P, AV_SAMPLE_FMT_FLT,

                                  2, 16, 8, "SSE2", ff_conv_flt_to_s16p_2ch_sse2);

        ff_audio_convert_set_func(ac, AV_SAMPLE_FMT_S16P, AV_SAMPLE_FMT_FLT,

                                  6, 16, 4, "SSE2", ff_conv_flt_to_s16p_6ch_sse2);

        ff_audio_convert_set_func(ac, AV_SAMPLE_FMT_FLTP, AV_SAMPLE_FMT_FLT,

                                  6, 16, 4, "SSE2", ff_conv_flt_to_fltp_6ch_sse2);

    }

    if (EXTERNAL_SSSE3(cpu_flags)) {

        ff_audio_convert_set_func(ac, AV_SAMPLE_FMT_FLT, AV_SAMPLE_FMT_S16P,

                                  6, 16, 4, "SSSE3", ff_conv_s16p_to_flt_6ch_ssse3);

        ff_audio_convert_set_func(ac, AV_SAMPLE_FMT_S16, AV_SAMPLE_FMT_FLTP,

                                  2, 16, 4, "SSSE3", ff_conv_fltp_to_s16_2ch_ssse3);

        ff_audio_convert_set_func(ac, AV_SAMPLE_FMT_S16P, AV_SAMPLE_FMT_S16,

                                  2, 16, 8, "SSSE3", ff_conv_s16_to_s16p_2ch_ssse3);

        ff_audio_convert_set_func(ac, AV_SAMPLE_FMT_S16P, AV_SAMPLE_FMT_S16,

                                  6, 16, 4, "SSSE3", ff_conv_s16_to_s16p_6ch_ssse3);

        ff_audio_convert_set_func(ac, AV_SAMPLE_FMT_FLTP, AV_SAMPLE_FMT_S16,

                                  6, 16, 4, "SSSE3", ff_conv_s16_to_fltp_6ch_ssse3);

        ff_audio_convert_set_func(ac, AV_SAMPLE_FMT_S16P, AV_SAMPLE_FMT_FLT,

                                  6, 16, 4, "SSSE3", ff_conv_flt_to_s16p_6ch_ssse3);

    }

    if (EXTERNAL_SSE4(cpu_flags)) {

        ff_audio_convert_set_func(ac, AV_SAMPLE_FMT_FLT, AV_SAMPLE_FMT_S16,

                                  0, 16, 8, "SSE4", ff_conv_s16_to_flt_sse4);

        ff_audio_convert_set_func(ac, AV_SAMPLE_FMT_FLT, AV_SAMPLE_FMT_FLTP,

                                  6, 16, 4, "SSE4", ff_conv_fltp_to_flt_6ch_sse4);

    }

    if (EXTERNAL_AVX(cpu_flags)) {

        ff_audio_convert_set_func(ac, AV_SAMPLE_FMT_FLT, AV_SAMPLE_FMT_S32,

                                  0, 32, 16, "AVX", ff_conv_s32_to_flt_avx);

        ff_audio_convert_set_func(ac, AV_SAMPLE_FMT_S32, AV_SAMPLE_FMT_FLT,

                                  0, 32, 32, "AVX", ff_conv_flt_to_s32_avx);

        ff_audio_convert_set_func(ac, AV_SAMPLE_FMT_S16, AV_SAMPLE_FMT_S16P,

                                  2, 16, 16, "AVX", ff_conv_s16p_to_s16_2ch_avx);

        ff_audio_convert_set_func(ac, AV_SAMPLE_FMT_S16, AV_SAMPLE_FMT_S16P,

                                  6, 16, 8, "AVX", ff_conv_s16p_to_s16_6ch_avx);

        ff_audio_convert_set_func(ac, AV_SAMPLE_FMT_FLT, AV_SAMPLE_FMT_S16P,

                                  2, 16, 8, "AVX", ff_conv_s16p_to_flt_2ch_avx);

        ff_audio_convert_set_func(ac, AV_SAMPLE_FMT_FLT, AV_SAMPLE_FMT_S16P,

                                  6, 16, 4, "AVX", ff_conv_s16p_to_flt_6ch_avx);

        ff_audio_convert_set_func(ac, AV_SAMPLE_FMT_S16, AV_SAMPLE_FMT_FLTP,

                                  6, 16, 4, "AVX", ff_conv_fltp_to_s16_6ch_avx);

        ff_audio_convert_set_func(ac, AV_SAMPLE_FMT_FLT, AV_SAMPLE_FMT_FLTP,

                                  6, 16, 4, "AVX", ff_conv_fltp_to_flt_6ch_avx);

        ff_audio_convert_set_func(ac, AV_SAMPLE_FMT_S16P, AV_SAMPLE_FMT_S16,

                                  2, 16, 8, "AVX", ff_conv_s16_to_s16p_2ch_avx);

        ff_audio_convert_set_func(ac, AV_SAMPLE_FMT_S16P, AV_SAMPLE_FMT_S16,

                                  6, 16, 4, "AVX", ff_conv_s16_to_s16p_6ch_avx);

        ff_audio_convert_set_func(ac, AV_SAMPLE_FMT_FLTP, AV_SAMPLE_FMT_S16,

                                  2, 16, 8, "AVX", ff_conv_s16_to_fltp_2ch_avx);

        ff_audio_convert_set_func(ac, AV_SAMPLE_FMT_FLTP, AV_SAMPLE_FMT_S16,

                                  6, 16, 4, "AVX", ff_conv_s16_to_fltp_6ch_avx);

        ff_audio_convert_set_func(ac, AV_SAMPLE_FMT_S16P, AV_SAMPLE_FMT_FLT,

                                  2, 16, 8, "AVX", ff_conv_flt_to_s16p_2ch_avx);

        ff_audio_convert_set_func(ac, AV_SAMPLE_FMT_S16P, AV_SAMPLE_FMT_FLT,

                                  6, 16, 4, "AVX", ff_conv_flt_to_s16p_6ch_avx);

        ff_audio_convert_set_func(ac, AV_SAMPLE_FMT_FLTP, AV_SAMPLE_FMT_FLT,

                                  2, 16, 4, "AVX", ff_conv_flt_to_fltp_2ch_avx);

        ff_audio_convert_set_func(ac, AV_SAMPLE_FMT_FLTP, AV_SAMPLE_FMT_FLT,

                                  6, 16, 4, "AVX", ff_conv_flt_to_fltp_6ch_avx);

    }

}
