av_cold void ff_vp8dsp_init(VP8DSPContext *dsp)

{

    dsp->vp8_luma_dc_wht    = vp8_luma_dc_wht_c;

    dsp->vp8_luma_dc_wht_dc = vp8_luma_dc_wht_dc_c;

    dsp->vp8_idct_add       = vp8_idct_add_c;

    dsp->vp8_idct_dc_add    = vp8_idct_dc_add_c;

    dsp->vp8_idct_dc_add4y  = vp8_idct_dc_add4y_c;

    dsp->vp8_idct_dc_add4uv = vp8_idct_dc_add4uv_c;



    dsp->vp8_v_loop_filter16y = vp8_v_loop_filter16_c;

    dsp->vp8_h_loop_filter16y = vp8_h_loop_filter16_c;

    dsp->vp8_v_loop_filter8uv = vp8_v_loop_filter8uv_c;

    dsp->vp8_h_loop_filter8uv = vp8_h_loop_filter8uv_c;



    dsp->vp8_v_loop_filter16y_inner = vp8_v_loop_filter16_inner_c;

    dsp->vp8_h_loop_filter16y_inner = vp8_h_loop_filter16_inner_c;

    dsp->vp8_v_loop_filter8uv_inner = vp8_v_loop_filter8uv_inner_c;

    dsp->vp8_h_loop_filter8uv_inner = vp8_h_loop_filter8uv_inner_c;



    dsp->vp8_v_loop_filter_simple = vp8_v_loop_filter_simple_c;

    dsp->vp8_h_loop_filter_simple = vp8_h_loop_filter_simple_c;



    VP8_MC_FUNC(0, 16);

    VP8_MC_FUNC(1, 8);

    VP8_MC_FUNC(2, 4);



    VP8_BILINEAR_MC_FUNC(0, 16);

    VP8_BILINEAR_MC_FUNC(1, 8);

    VP8_BILINEAR_MC_FUNC(2, 4);



    if (ARCH_ARM)

        ff_vp8dsp_init_arm(dsp);

    if (ARCH_PPC)

        ff_vp8dsp_init_ppc(dsp);

    if (ARCH_X86)

        ff_vp8dsp_init_x86(dsp);

}
