static av_cold void h264dsp_init_neon(H264DSPContext *c, const int bit_depth,

                                      const int chroma_format_idc)

{

#if HAVE_NEON

    if (bit_depth == 8) {

        c->h264_v_loop_filter_luma   = ff_h264_v_loop_filter_luma_neon;

        c->h264_h_loop_filter_luma   = ff_h264_h_loop_filter_luma_neon;

        if(chroma_format_idc == 1){

        c->h264_v_loop_filter_chroma = ff_h264_v_loop_filter_chroma_neon;

        c->h264_h_loop_filter_chroma = ff_h264_h_loop_filter_chroma_neon;

        }



        c->weight_h264_pixels_tab[0] = ff_weight_h264_pixels_16_neon;

        c->weight_h264_pixels_tab[1] = ff_weight_h264_pixels_8_neon;

        c->weight_h264_pixels_tab[2] = ff_weight_h264_pixels_4_neon;



        c->biweight_h264_pixels_tab[0] = ff_biweight_h264_pixels_16_neon;

        c->biweight_h264_pixels_tab[1] = ff_biweight_h264_pixels_8_neon;

        c->biweight_h264_pixels_tab[2] = ff_biweight_h264_pixels_4_neon;



        c->h264_idct_add        = ff_h264_idct_add_neon;

        c->h264_idct_dc_add     = ff_h264_idct_dc_add_neon;

        c->h264_idct_add16      = ff_h264_idct_add16_neon;

        c->h264_idct_add16intra = ff_h264_idct_add16intra_neon;

        if (chroma_format_idc <= 1)

            c->h264_idct_add8   = ff_h264_idct_add8_neon;

        c->h264_idct8_add       = ff_h264_idct8_add_neon;

        c->h264_idct8_dc_add    = ff_h264_idct8_dc_add_neon;

        c->h264_idct8_add4      = ff_h264_idct8_add4_neon;

    }

#endif // HAVE_NEON

}
