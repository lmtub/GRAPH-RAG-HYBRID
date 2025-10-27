void ff_vc1dsp_init_mmx(VC1DSPContext *dsp)

{

    int mm_flags = av_get_cpu_flags();



#if HAVE_INLINE_ASM

    if (mm_flags & AV_CPU_FLAG_MMX) {

        dsp->put_vc1_mspel_pixels_tab[ 0] = ff_put_vc1_mspel_mc00_mmx;

        dsp->put_vc1_mspel_pixels_tab[ 4] = put_vc1_mspel_mc01_mmx;

        dsp->put_vc1_mspel_pixels_tab[ 8] = put_vc1_mspel_mc02_mmx;

        dsp->put_vc1_mspel_pixels_tab[12] = put_vc1_mspel_mc03_mmx;



        dsp->put_vc1_mspel_pixels_tab[ 1] = put_vc1_mspel_mc10_mmx;

        dsp->put_vc1_mspel_pixels_tab[ 5] = put_vc1_mspel_mc11_mmx;

        dsp->put_vc1_mspel_pixels_tab[ 9] = put_vc1_mspel_mc12_mmx;

        dsp->put_vc1_mspel_pixels_tab[13] = put_vc1_mspel_mc13_mmx;



        dsp->put_vc1_mspel_pixels_tab[ 2] = put_vc1_mspel_mc20_mmx;

        dsp->put_vc1_mspel_pixels_tab[ 6] = put_vc1_mspel_mc21_mmx;

        dsp->put_vc1_mspel_pixels_tab[10] = put_vc1_mspel_mc22_mmx;

        dsp->put_vc1_mspel_pixels_tab[14] = put_vc1_mspel_mc23_mmx;



        dsp->put_vc1_mspel_pixels_tab[ 3] = put_vc1_mspel_mc30_mmx;

        dsp->put_vc1_mspel_pixels_tab[ 7] = put_vc1_mspel_mc31_mmx;

        dsp->put_vc1_mspel_pixels_tab[11] = put_vc1_mspel_mc32_mmx;

        dsp->put_vc1_mspel_pixels_tab[15] = put_vc1_mspel_mc33_mmx;

    }



    if (mm_flags & AV_CPU_FLAG_MMX2){

        dsp->avg_vc1_mspel_pixels_tab[ 0] = ff_avg_vc1_mspel_mc00_mmx2;

        dsp->avg_vc1_mspel_pixels_tab[ 4] = avg_vc1_mspel_mc01_mmx2;

        dsp->avg_vc1_mspel_pixels_tab[ 8] = avg_vc1_mspel_mc02_mmx2;

        dsp->avg_vc1_mspel_pixels_tab[12] = avg_vc1_mspel_mc03_mmx2;



        dsp->avg_vc1_mspel_pixels_tab[ 1] = avg_vc1_mspel_mc10_mmx2;

        dsp->avg_vc1_mspel_pixels_tab[ 5] = avg_vc1_mspel_mc11_mmx2;

        dsp->avg_vc1_mspel_pixels_tab[ 9] = avg_vc1_mspel_mc12_mmx2;

        dsp->avg_vc1_mspel_pixels_tab[13] = avg_vc1_mspel_mc13_mmx2;



        dsp->avg_vc1_mspel_pixels_tab[ 2] = avg_vc1_mspel_mc20_mmx2;

        dsp->avg_vc1_mspel_pixels_tab[ 6] = avg_vc1_mspel_mc21_mmx2;

        dsp->avg_vc1_mspel_pixels_tab[10] = avg_vc1_mspel_mc22_mmx2;

        dsp->avg_vc1_mspel_pixels_tab[14] = avg_vc1_mspel_mc23_mmx2;



        dsp->avg_vc1_mspel_pixels_tab[ 3] = avg_vc1_mspel_mc30_mmx2;

        dsp->avg_vc1_mspel_pixels_tab[ 7] = avg_vc1_mspel_mc31_mmx2;

        dsp->avg_vc1_mspel_pixels_tab[11] = avg_vc1_mspel_mc32_mmx2;

        dsp->avg_vc1_mspel_pixels_tab[15] = avg_vc1_mspel_mc33_mmx2;



        dsp->vc1_inv_trans_8x8_dc = vc1_inv_trans_8x8_dc_mmx2;

        dsp->vc1_inv_trans_4x8_dc = vc1_inv_trans_4x8_dc_mmx2;

        dsp->vc1_inv_trans_8x4_dc = vc1_inv_trans_8x4_dc_mmx2;

        dsp->vc1_inv_trans_4x4_dc = vc1_inv_trans_4x4_dc_mmx2;

    }

#endif /* HAVE_INLINE_ASM */



#define ASSIGN_LF(EXT) \

        dsp->vc1_v_loop_filter4  = ff_vc1_v_loop_filter4_ ## EXT; \

        dsp->vc1_h_loop_filter4  = ff_vc1_h_loop_filter4_ ## EXT; \

        dsp->vc1_v_loop_filter8  = ff_vc1_v_loop_filter8_ ## EXT; \

        dsp->vc1_h_loop_filter8  = ff_vc1_h_loop_filter8_ ## EXT; \

        dsp->vc1_v_loop_filter16 = vc1_v_loop_filter16_ ## EXT; \

        dsp->vc1_h_loop_filter16 = vc1_h_loop_filter16_ ## EXT



#if HAVE_YASM

    if (mm_flags & AV_CPU_FLAG_MMX) {

        dsp->put_no_rnd_vc1_chroma_pixels_tab[0]= ff_put_vc1_chroma_mc8_mmx_nornd;

    }

    return;

    if (mm_flags & AV_CPU_FLAG_MMX2) {

        ASSIGN_LF(mmx2);

        dsp->avg_no_rnd_vc1_chroma_pixels_tab[0]= ff_avg_vc1_chroma_mc8_mmx2_nornd;

    } else if (mm_flags & AV_CPU_FLAG_3DNOW) {

        dsp->avg_no_rnd_vc1_chroma_pixels_tab[0]= ff_avg_vc1_chroma_mc8_3dnow_nornd;

    }



    if (mm_flags & AV_CPU_FLAG_SSE2) {

        dsp->vc1_v_loop_filter8  = ff_vc1_v_loop_filter8_sse2;

        dsp->vc1_h_loop_filter8  = ff_vc1_h_loop_filter8_sse2;

        dsp->vc1_v_loop_filter16 = vc1_v_loop_filter16_sse2;

        dsp->vc1_h_loop_filter16 = vc1_h_loop_filter16_sse2;

    }

    if (mm_flags & AV_CPU_FLAG_SSSE3) {

        ASSIGN_LF(ssse3);

        dsp->put_no_rnd_vc1_chroma_pixels_tab[0]= ff_put_vc1_chroma_mc8_ssse3_nornd;

        dsp->avg_no_rnd_vc1_chroma_pixels_tab[0]= ff_avg_vc1_chroma_mc8_ssse3_nornd;

    }

    if (mm_flags & AV_CPU_FLAG_SSE4) {

        dsp->vc1_h_loop_filter8  = ff_vc1_h_loop_filter8_sse4;

        dsp->vc1_h_loop_filter16 = vc1_h_loop_filter16_sse4;

    }

#endif

}
