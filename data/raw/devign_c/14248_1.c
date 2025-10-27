LF_FUNC (h,  luma,         sse2)

LF_IFUNC(h,  luma_intra,   sse2)

LF_FUNC (v,  luma,         sse2)

LF_IFUNC(v,  luma_intra,   sse2)



/***********************************/

/* weighted prediction */



#define H264_WEIGHT(W, H, OPT) \

void ff_h264_weight_ ## W ## x ## H ## _ ## OPT(uint8_t *dst, \

    int stride, int log2_denom, int weight, int offset);



#define H264_BIWEIGHT(W, H, OPT) \

void ff_h264_biweight_ ## W ## x ## H ## _ ## OPT(uint8_t *dst, \

    uint8_t *src, int stride, int log2_denom, int weightd, \

    int weights, int offset);



#define H264_BIWEIGHT_MMX(W,H) \

H264_WEIGHT  (W, H, mmx2) \

H264_BIWEIGHT(W, H, mmx2)



#define H264_BIWEIGHT_MMX_SSE(W,H) \

H264_BIWEIGHT_MMX(W, H) \

H264_WEIGHT      (W, H, sse2) \

H264_BIWEIGHT    (W, H, sse2) \

H264_BIWEIGHT    (W, H, ssse3)



H264_BIWEIGHT_MMX_SSE(16, 16)

H264_BIWEIGHT_MMX_SSE(16,  8)

H264_BIWEIGHT_MMX_SSE( 8, 16)

H264_BIWEIGHT_MMX_SSE( 8,  8)

H264_BIWEIGHT_MMX_SSE( 8,  4)

H264_BIWEIGHT_MMX    ( 4,  8)

H264_BIWEIGHT_MMX    ( 4,  4)

H264_BIWEIGHT_MMX    ( 4,  2)



void ff_h264dsp_init_x86(H264DSPContext *c)

{

    int mm_flags = av_get_cpu_flags();



    if (mm_flags & AV_CPU_FLAG_MMX2) {

        c->h264_loop_filter_strength= h264_loop_filter_strength_mmx2;

    }

#if HAVE_YASM

    if (mm_flags & AV_CPU_FLAG_MMX) {

        c->h264_idct_dc_add=

        c->h264_idct_add= ff_h264_idct_add_mmx;

        c->h264_idct8_dc_add=

        c->h264_idct8_add= ff_h264_idct8_add_mmx;



        c->h264_idct_add16     = ff_h264_idct_add16_mmx;

        c->h264_idct8_add4     = ff_h264_idct8_add4_mmx;

        c->h264_idct_add8      = ff_h264_idct_add8_mmx;

        c->h264_idct_add16intra= ff_h264_idct_add16intra_mmx;



        if (mm_flags & AV_CPU_FLAG_MMX2) {

            c->h264_idct_dc_add= ff_h264_idct_dc_add_mmx2;

            c->h264_idct8_dc_add= ff_h264_idct8_dc_add_mmx2;

            c->h264_idct_add16     = ff_h264_idct_add16_mmx2;

            c->h264_idct8_add4     = ff_h264_idct8_add4_mmx2;

            c->h264_idct_add8      = ff_h264_idct_add8_mmx2;

            c->h264_idct_add16intra= ff_h264_idct_add16intra_mmx2;



            c->h264_v_loop_filter_chroma= ff_x264_deblock_v_chroma_mmxext;

            c->h264_h_loop_filter_chroma= ff_x264_deblock_h_chroma_mmxext;

            c->h264_v_loop_filter_chroma_intra= ff_x264_deblock_v_chroma_intra_mmxext;

            c->h264_h_loop_filter_chroma_intra= ff_x264_deblock_h_chroma_intra_mmxext;

#if ARCH_X86_32

            c->h264_v_loop_filter_luma= ff_x264_deblock_v_luma_mmxext;

            c->h264_h_loop_filter_luma= ff_x264_deblock_h_luma_mmxext;

            c->h264_v_loop_filter_luma_intra = ff_x264_deblock_v_luma_intra_mmxext;

            c->h264_h_loop_filter_luma_intra = ff_x264_deblock_h_luma_intra_mmxext;


            c->weight_h264_pixels_tab[0]= ff_h264_weight_16x16_mmx2;

            c->weight_h264_pixels_tab[1]= ff_h264_weight_16x8_mmx2;

            c->weight_h264_pixels_tab[2]= ff_h264_weight_8x16_mmx2;

            c->weight_h264_pixels_tab[3]= ff_h264_weight_8x8_mmx2;

            c->weight_h264_pixels_tab[4]= ff_h264_weight_8x4_mmx2;

            c->weight_h264_pixels_tab[5]= ff_h264_weight_4x8_mmx2;

            c->weight_h264_pixels_tab[6]= ff_h264_weight_4x4_mmx2;

            c->weight_h264_pixels_tab[7]= ff_h264_weight_4x2_mmx2;



            c->biweight_h264_pixels_tab[0]= ff_h264_biweight_16x16_mmx2;

            c->biweight_h264_pixels_tab[1]= ff_h264_biweight_16x8_mmx2;

            c->biweight_h264_pixels_tab[2]= ff_h264_biweight_8x16_mmx2;

            c->biweight_h264_pixels_tab[3]= ff_h264_biweight_8x8_mmx2;

            c->biweight_h264_pixels_tab[4]= ff_h264_biweight_8x4_mmx2;

            c->biweight_h264_pixels_tab[5]= ff_h264_biweight_4x8_mmx2;

            c->biweight_h264_pixels_tab[6]= ff_h264_biweight_4x4_mmx2;

            c->biweight_h264_pixels_tab[7]= ff_h264_biweight_4x2_mmx2;



            if (mm_flags&AV_CPU_FLAG_SSE2) {

                c->h264_idct8_add = ff_h264_idct8_add_sse2;

                c->h264_idct8_add4= ff_h264_idct8_add4_sse2;



                c->weight_h264_pixels_tab[0]= ff_h264_weight_16x16_sse2;

                c->weight_h264_pixels_tab[1]= ff_h264_weight_16x8_sse2;

                c->weight_h264_pixels_tab[2]= ff_h264_weight_8x16_sse2;

                c->weight_h264_pixels_tab[3]= ff_h264_weight_8x8_sse2;

                c->weight_h264_pixels_tab[4]= ff_h264_weight_8x4_sse2;



                c->biweight_h264_pixels_tab[0]= ff_h264_biweight_16x16_sse2;

                c->biweight_h264_pixels_tab[1]= ff_h264_biweight_16x8_sse2;

                c->biweight_h264_pixels_tab[2]= ff_h264_biweight_8x16_sse2;

                c->biweight_h264_pixels_tab[3]= ff_h264_biweight_8x8_sse2;

                c->biweight_h264_pixels_tab[4]= ff_h264_biweight_8x4_sse2;




                c->h264_v_loop_filter_luma = ff_x264_deblock_v_luma_sse2;

                c->h264_h_loop_filter_luma = ff_x264_deblock_h_luma_sse2;

                c->h264_v_loop_filter_luma_intra = ff_x264_deblock_v_luma_intra_sse2;

                c->h264_h_loop_filter_luma_intra = ff_x264_deblock_h_luma_intra_sse2;




                c->h264_idct_add16 = ff_h264_idct_add16_sse2;

                c->h264_idct_add8  = ff_h264_idct_add8_sse2;

                c->h264_idct_add16intra = ff_h264_idct_add16intra_sse2;

            }

            if (mm_flags&AV_CPU_FLAG_SSSE3) {

                c->biweight_h264_pixels_tab[0]= ff_h264_biweight_16x16_ssse3;

                c->biweight_h264_pixels_tab[1]= ff_h264_biweight_16x8_ssse3;

                c->biweight_h264_pixels_tab[2]= ff_h264_biweight_8x16_ssse3;

                c->biweight_h264_pixels_tab[3]= ff_h264_biweight_8x8_ssse3;

                c->biweight_h264_pixels_tab[4]= ff_h264_biweight_8x4_ssse3;

            }

        }

    }


}