PROTO4(_pack_2ch_)

PROTO4(_pack_6ch_)

PROTO4(_pack_8ch_)

PROTO4(_unpack_2ch_)

PROTO4(_unpack_6ch_)



av_cold void swri_audio_convert_init_x86(struct AudioConvert *ac,

                                 enum AVSampleFormat out_fmt,

                                 enum AVSampleFormat in_fmt,

                                 int channels){

    int mm_flags = av_get_cpu_flags();



    ac->simd_f= NULL;



//FIXME add memcpy case



#define MULTI_CAPS_FUNC(flag, cap) \

    if (EXTERNAL_##flag(mm_flags)) {\

        if(   out_fmt == AV_SAMPLE_FMT_S32  && in_fmt == AV_SAMPLE_FMT_S16 || out_fmt == AV_SAMPLE_FMT_S32P && in_fmt == AV_SAMPLE_FMT_S16P)\

            ac->simd_f =  ff_int16_to_int32_a_ ## cap;\

        if(   out_fmt == AV_SAMPLE_FMT_S16  && in_fmt == AV_SAMPLE_FMT_S32 || out_fmt == AV_SAMPLE_FMT_S16P && in_fmt == AV_SAMPLE_FMT_S32P)\

            ac->simd_f =  ff_int32_to_int16_a_ ## cap;\

    }



MULTI_CAPS_FUNC(MMX, mmx)

MULTI_CAPS_FUNC(SSE2, sse2)



    if(EXTERNAL_MMX(mm_flags)) {

        if(channels == 6) {

            if(   out_fmt == AV_SAMPLE_FMT_FLT  && in_fmt == AV_SAMPLE_FMT_FLTP || out_fmt == AV_SAMPLE_FMT_S32 && in_fmt == AV_SAMPLE_FMT_S32P)

                ac->simd_f =  ff_pack_6ch_float_to_float_a_mmx;

        }

    }

    if(EXTERNAL_SSE(mm_flags)) {

        if(channels == 6) {

            if(   out_fmt == AV_SAMPLE_FMT_FLT  && in_fmt == AV_SAMPLE_FMT_FLTP || out_fmt == AV_SAMPLE_FMT_S32 && in_fmt == AV_SAMPLE_FMT_S32P)

                ac->simd_f =  ff_pack_6ch_float_to_float_a_sse;



            if(   out_fmt == AV_SAMPLE_FMT_FLTP  && in_fmt == AV_SAMPLE_FMT_FLT || out_fmt == AV_SAMPLE_FMT_S32P && in_fmt == AV_SAMPLE_FMT_S32)

                ac->simd_f =  ff_unpack_6ch_float_to_float_a_sse;

        }

    }

    if(EXTERNAL_SSE2(mm_flags)) {

        if(   out_fmt == AV_SAMPLE_FMT_FLT  && in_fmt == AV_SAMPLE_FMT_S32 || out_fmt == AV_SAMPLE_FMT_FLTP && in_fmt == AV_SAMPLE_FMT_S32P)

            ac->simd_f =  ff_int32_to_float_a_sse2;

        if(   out_fmt == AV_SAMPLE_FMT_FLT  && in_fmt == AV_SAMPLE_FMT_S16 || out_fmt == AV_SAMPLE_FMT_FLTP && in_fmt == AV_SAMPLE_FMT_S16P)

            ac->simd_f =  ff_int16_to_float_a_sse2;

        if(   out_fmt == AV_SAMPLE_FMT_S32  && in_fmt == AV_SAMPLE_FMT_FLT || out_fmt == AV_SAMPLE_FMT_S32P && in_fmt == AV_SAMPLE_FMT_FLTP)

            ac->simd_f =  ff_float_to_int32_a_sse2;

        if(   out_fmt == AV_SAMPLE_FMT_S16  && in_fmt == AV_SAMPLE_FMT_FLT || out_fmt == AV_SAMPLE_FMT_S16P && in_fmt == AV_SAMPLE_FMT_FLTP)

            ac->simd_f =  ff_float_to_int16_a_sse2;



        if(channels == 2) {

            if(   out_fmt == AV_SAMPLE_FMT_FLT  && in_fmt == AV_SAMPLE_FMT_FLTP || out_fmt == AV_SAMPLE_FMT_S32 && in_fmt == AV_SAMPLE_FMT_S32P)

                ac->simd_f =  ff_pack_2ch_int32_to_int32_a_sse2;

            if(   out_fmt == AV_SAMPLE_FMT_S16  && in_fmt == AV_SAMPLE_FMT_S16P)

                ac->simd_f =  ff_pack_2ch_int16_to_int16_a_sse2;

            if(   out_fmt == AV_SAMPLE_FMT_S32  && in_fmt == AV_SAMPLE_FMT_S16P)

                ac->simd_f =  ff_pack_2ch_int16_to_int32_a_sse2;

            if(   out_fmt == AV_SAMPLE_FMT_S16  && in_fmt == AV_SAMPLE_FMT_S32P)

                ac->simd_f =  ff_pack_2ch_int32_to_int16_a_sse2;



            if(   out_fmt == AV_SAMPLE_FMT_FLTP  && in_fmt == AV_SAMPLE_FMT_FLT || out_fmt == AV_SAMPLE_FMT_S32P && in_fmt == AV_SAMPLE_FMT_S32)

                ac->simd_f =  ff_unpack_2ch_int32_to_int32_a_sse2;

            if(   out_fmt == AV_SAMPLE_FMT_S16P  && in_fmt == AV_SAMPLE_FMT_S16)

                ac->simd_f =  ff_unpack_2ch_int16_to_int16_a_sse2;

            if(   out_fmt == AV_SAMPLE_FMT_S32P  && in_fmt == AV_SAMPLE_FMT_S16)

                ac->simd_f =  ff_unpack_2ch_int16_to_int32_a_sse2;

            if(   out_fmt == AV_SAMPLE_FMT_S16P  && in_fmt == AV_SAMPLE_FMT_S32)

                ac->simd_f =  ff_unpack_2ch_int32_to_int16_a_sse2;



            if(   out_fmt == AV_SAMPLE_FMT_FLT  && in_fmt == AV_SAMPLE_FMT_S32P)

                ac->simd_f =  ff_pack_2ch_int32_to_float_a_sse2;

            if(   out_fmt == AV_SAMPLE_FMT_S32  && in_fmt == AV_SAMPLE_FMT_FLTP)

                ac->simd_f =  ff_pack_2ch_float_to_int32_a_sse2;

            if(   out_fmt == AV_SAMPLE_FMT_FLT  && in_fmt == AV_SAMPLE_FMT_S16P)

                ac->simd_f =  ff_pack_2ch_int16_to_float_a_sse2;

            if(   out_fmt == AV_SAMPLE_FMT_S16  && in_fmt == AV_SAMPLE_FMT_FLTP)

                ac->simd_f =  ff_pack_2ch_float_to_int16_a_sse2;

            if(   out_fmt == AV_SAMPLE_FMT_FLTP  && in_fmt == AV_SAMPLE_FMT_S32)

                ac->simd_f =  ff_unpack_2ch_int32_to_float_a_sse2;

            if(   out_fmt == AV_SAMPLE_FMT_S32P  && in_fmt == AV_SAMPLE_FMT_FLT)

                ac->simd_f =  ff_unpack_2ch_float_to_int32_a_sse2;

            if(   out_fmt == AV_SAMPLE_FMT_FLTP  && in_fmt == AV_SAMPLE_FMT_S16)

                ac->simd_f =  ff_unpack_2ch_int16_to_float_a_sse2;

            if(   out_fmt == AV_SAMPLE_FMT_S16P  && in_fmt == AV_SAMPLE_FMT_FLT)

                ac->simd_f =  ff_unpack_2ch_float_to_int16_a_sse2;

        }

        if(channels == 6) {

            if(   out_fmt == AV_SAMPLE_FMT_FLT  && in_fmt == AV_SAMPLE_FMT_S32P)

                ac->simd_f =  ff_pack_6ch_int32_to_float_a_sse2;

            if(   out_fmt == AV_SAMPLE_FMT_S32  && in_fmt == AV_SAMPLE_FMT_FLTP)

                ac->simd_f =  ff_pack_6ch_float_to_int32_a_sse2;



            if(   out_fmt == AV_SAMPLE_FMT_FLTP  && in_fmt == AV_SAMPLE_FMT_S32)

                ac->simd_f =  ff_unpack_6ch_int32_to_float_a_sse2;

            if(   out_fmt == AV_SAMPLE_FMT_S32P  && in_fmt == AV_SAMPLE_FMT_FLT)

                ac->simd_f =  ff_unpack_6ch_float_to_int32_a_sse2;

        }

        if(channels == 8) {

            if(   out_fmt == AV_SAMPLE_FMT_FLT  && in_fmt == AV_SAMPLE_FMT_FLTP || out_fmt == AV_SAMPLE_FMT_S32 && in_fmt == AV_SAMPLE_FMT_S32P)

                ac->simd_f =  ff_pack_8ch_float_to_float_a_sse2;

            if(   out_fmt == AV_SAMPLE_FMT_FLT  && in_fmt == AV_SAMPLE_FMT_S32P)

                ac->simd_f =  ff_pack_8ch_int32_to_float_a_sse2;

            if(   out_fmt == AV_SAMPLE_FMT_S32  && in_fmt == AV_SAMPLE_FMT_FLTP)

                ac->simd_f =  ff_pack_8ch_float_to_int32_a_sse2;

        }

    }

    if(EXTERNAL_SSSE3(mm_flags)) {

        if(channels == 2) {

            if(   out_fmt == AV_SAMPLE_FMT_S16P  && in_fmt == AV_SAMPLE_FMT_S16)

                ac->simd_f =  ff_unpack_2ch_int16_to_int16_a_ssse3;

            if(   out_fmt == AV_SAMPLE_FMT_S32P  && in_fmt == AV_SAMPLE_FMT_S16)

                ac->simd_f =  ff_unpack_2ch_int16_to_int32_a_ssse3;

            if(   out_fmt == AV_SAMPLE_FMT_FLTP  && in_fmt == AV_SAMPLE_FMT_S16)

                ac->simd_f =  ff_unpack_2ch_int16_to_float_a_ssse3;

        }

    }

    if(EXTERNAL_AVX(mm_flags)) {

        if(   out_fmt == AV_SAMPLE_FMT_FLT  && in_fmt == AV_SAMPLE_FMT_S32 || out_fmt == AV_SAMPLE_FMT_FLTP && in_fmt == AV_SAMPLE_FMT_S32P)

            ac->simd_f =  ff_int32_to_float_a_avx;

        if(channels == 6) {

            if(   out_fmt == AV_SAMPLE_FMT_FLT  && in_fmt == AV_SAMPLE_FMT_FLTP || out_fmt == AV_SAMPLE_FMT_S32 && in_fmt == AV_SAMPLE_FMT_S32P)

                ac->simd_f =  ff_pack_6ch_float_to_float_a_avx;

            if(   out_fmt == AV_SAMPLE_FMT_FLT  && in_fmt == AV_SAMPLE_FMT_S32P)

                ac->simd_f =  ff_pack_6ch_int32_to_float_a_avx;

            if(   out_fmt == AV_SAMPLE_FMT_S32  && in_fmt == AV_SAMPLE_FMT_FLTP)

                ac->simd_f =  ff_pack_6ch_float_to_int32_a_avx;



            if(   out_fmt == AV_SAMPLE_FMT_FLTP  && in_fmt == AV_SAMPLE_FMT_FLT || out_fmt == AV_SAMPLE_FMT_S32P && in_fmt == AV_SAMPLE_FMT_S32)

                ac->simd_f =  ff_unpack_6ch_float_to_float_a_avx;

            if(   out_fmt == AV_SAMPLE_FMT_FLTP  && in_fmt == AV_SAMPLE_FMT_S32)

                ac->simd_f =  ff_unpack_6ch_int32_to_float_a_avx;

            if(   out_fmt == AV_SAMPLE_FMT_S32P  && in_fmt == AV_SAMPLE_FMT_FLT)

                ac->simd_f =  ff_unpack_6ch_float_to_int32_a_avx;

        }

        if(channels == 8) {

            if(   out_fmt == AV_SAMPLE_FMT_FLT  && in_fmt == AV_SAMPLE_FMT_FLTP || out_fmt == AV_SAMPLE_FMT_S32 && in_fmt == AV_SAMPLE_FMT_S32P)

                ac->simd_f =  ff_pack_8ch_float_to_float_a_avx;

            if(   out_fmt == AV_SAMPLE_FMT_FLT  && in_fmt == AV_SAMPLE_FMT_S32P)

                ac->simd_f =  ff_pack_8ch_int32_to_float_a_avx;

            if(   out_fmt == AV_SAMPLE_FMT_S32  && in_fmt == AV_SAMPLE_FMT_FLTP)

                ac->simd_f =  ff_pack_8ch_float_to_int32_a_avx;

        }

    }

    if(EXTERNAL_AVX2(mm_flags)) {

        if(   out_fmt == AV_SAMPLE_FMT_S32  && in_fmt == AV_SAMPLE_FMT_FLT || out_fmt == AV_SAMPLE_FMT_S32P && in_fmt == AV_SAMPLE_FMT_FLTP)

            ac->simd_f =  ff_float_to_int32_a_avx2;

    }

}
