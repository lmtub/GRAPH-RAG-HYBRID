av_cold void ff_vp8dsp_init_armv6(VP8DSPContext *dsp)

{

    dsp->vp8_luma_dc_wht    = ff_vp8_luma_dc_wht_armv6;

    dsp->vp8_luma_dc_wht_dc = ff_vp8_luma_dc_wht_dc_armv6;



    dsp->vp8_idct_add       = ff_vp8_idct_add_armv6;

    dsp->vp8_idct_dc_add    = ff_vp8_idct_dc_add_armv6;

    dsp->vp8_idct_dc_add4y  = ff_vp8_idct_dc_add4y_armv6;

    dsp->vp8_idct_dc_add4uv = ff_vp8_idct_dc_add4uv_armv6;



    dsp->vp8_v_loop_filter16y = ff_vp8_v_loop_filter16_armv6;

    dsp->vp8_h_loop_filter16y = ff_vp8_h_loop_filter16_armv6;

    dsp->vp8_v_loop_filter8uv = ff_vp8_v_loop_filter8uv_armv6;

    dsp->vp8_h_loop_filter8uv = ff_vp8_h_loop_filter8uv_armv6;



    dsp->vp8_v_loop_filter16y_inner = ff_vp8_v_loop_filter16_inner_armv6;

    dsp->vp8_h_loop_filter16y_inner = ff_vp8_h_loop_filter16_inner_armv6;

    dsp->vp8_v_loop_filter8uv_inner = ff_vp8_v_loop_filter8uv_inner_armv6;

    dsp->vp8_h_loop_filter8uv_inner = ff_vp8_h_loop_filter8uv_inner_armv6;



    dsp->vp8_v_loop_filter_simple = ff_vp8_v_loop_filter16_simple_armv6;

    dsp->vp8_h_loop_filter_simple = ff_vp8_h_loop_filter16_simple_armv6;



    dsp->put_vp8_epel_pixels_tab[0][0][0] = ff_put_vp8_pixels16_armv6;

    dsp->put_vp8_epel_pixels_tab[0][0][2] = ff_put_vp8_epel16_h6_armv6;

    dsp->put_vp8_epel_pixels_tab[0][2][0] = ff_put_vp8_epel16_v6_armv6;

    dsp->put_vp8_epel_pixels_tab[0][2][2] = ff_put_vp8_epel16_h6v6_armv6;



    dsp->put_vp8_epel_pixels_tab[1][0][0] = ff_put_vp8_pixels8_armv6;

    dsp->put_vp8_epel_pixels_tab[1][0][1] = ff_put_vp8_epel8_h4_armv6;

    dsp->put_vp8_epel_pixels_tab[1][0][2] = ff_put_vp8_epel8_h6_armv6;

    dsp->put_vp8_epel_pixels_tab[1][1][0] = ff_put_vp8_epel8_v4_armv6;

    dsp->put_vp8_epel_pixels_tab[1][1][1] = ff_put_vp8_epel8_h4v4_armv6;

    dsp->put_vp8_epel_pixels_tab[1][1][2] = ff_put_vp8_epel8_h6v4_armv6;

    dsp->put_vp8_epel_pixels_tab[1][2][0] = ff_put_vp8_epel8_v6_armv6;

    dsp->put_vp8_epel_pixels_tab[1][2][1] = ff_put_vp8_epel8_h4v6_armv6;

    dsp->put_vp8_epel_pixels_tab[1][2][2] = ff_put_vp8_epel8_h6v6_armv6;



    dsp->put_vp8_epel_pixels_tab[2][0][0] = ff_put_vp8_pixels4_armv6;

    dsp->put_vp8_epel_pixels_tab[2][0][1] = ff_put_vp8_epel4_h4_armv6;

    dsp->put_vp8_epel_pixels_tab[2][0][2] = ff_put_vp8_epel4_h6_armv6;

    dsp->put_vp8_epel_pixels_tab[2][1][0] = ff_put_vp8_epel4_v4_armv6;

    dsp->put_vp8_epel_pixels_tab[2][1][1] = ff_put_vp8_epel4_h4v4_armv6;

    dsp->put_vp8_epel_pixels_tab[2][1][2] = ff_put_vp8_epel4_h6v4_armv6;

    dsp->put_vp8_epel_pixels_tab[2][2][0] = ff_put_vp8_epel4_v6_armv6;

    dsp->put_vp8_epel_pixels_tab[2][2][1] = ff_put_vp8_epel4_h4v6_armv6;

    dsp->put_vp8_epel_pixels_tab[2][2][2] = ff_put_vp8_epel4_h6v6_armv6;



    dsp->put_vp8_bilinear_pixels_tab[0][0][0] = ff_put_vp8_pixels16_armv6;

    dsp->put_vp8_bilinear_pixels_tab[0][0][1] = ff_put_vp8_bilin16_h_armv6;

    dsp->put_vp8_bilinear_pixels_tab[0][0][2] = ff_put_vp8_bilin16_h_armv6;

    dsp->put_vp8_bilinear_pixels_tab[0][1][0] = ff_put_vp8_bilin16_v_armv6;

    dsp->put_vp8_bilinear_pixels_tab[0][1][1] = ff_put_vp8_bilin16_hv_armv6;

    dsp->put_vp8_bilinear_pixels_tab[0][1][2] = ff_put_vp8_bilin16_hv_armv6;

    dsp->put_vp8_bilinear_pixels_tab[0][2][0] = ff_put_vp8_bilin16_v_armv6;

    dsp->put_vp8_bilinear_pixels_tab[0][2][1] = ff_put_vp8_bilin16_hv_armv6;

    dsp->put_vp8_bilinear_pixels_tab[0][2][2] = ff_put_vp8_bilin16_hv_armv6;



    dsp->put_vp8_bilinear_pixels_tab[1][0][0] = ff_put_vp8_pixels8_armv6;

    dsp->put_vp8_bilinear_pixels_tab[1][0][1] = ff_put_vp8_bilin8_h_armv6;

    dsp->put_vp8_bilinear_pixels_tab[1][0][2] = ff_put_vp8_bilin8_h_armv6;

    dsp->put_vp8_bilinear_pixels_tab[1][1][0] = ff_put_vp8_bilin8_v_armv6;

    dsp->put_vp8_bilinear_pixels_tab[1][1][1] = ff_put_vp8_bilin8_hv_armv6;

    dsp->put_vp8_bilinear_pixels_tab[1][1][2] = ff_put_vp8_bilin8_hv_armv6;

    dsp->put_vp8_bilinear_pixels_tab[1][2][0] = ff_put_vp8_bilin8_v_armv6;

    dsp->put_vp8_bilinear_pixels_tab[1][2][1] = ff_put_vp8_bilin8_hv_armv6;

    dsp->put_vp8_bilinear_pixels_tab[1][2][2] = ff_put_vp8_bilin8_hv_armv6;



    dsp->put_vp8_bilinear_pixels_tab[2][0][0] = ff_put_vp8_pixels4_armv6;

    dsp->put_vp8_bilinear_pixels_tab[2][0][1] = ff_put_vp8_bilin4_h_armv6;

    dsp->put_vp8_bilinear_pixels_tab[2][0][2] = ff_put_vp8_bilin4_h_armv6;

    dsp->put_vp8_bilinear_pixels_tab[2][1][0] = ff_put_vp8_bilin4_v_armv6;

    dsp->put_vp8_bilinear_pixels_tab[2][1][1] = ff_put_vp8_bilin4_hv_armv6;

    dsp->put_vp8_bilinear_pixels_tab[2][1][2] = ff_put_vp8_bilin4_hv_armv6;

    dsp->put_vp8_bilinear_pixels_tab[2][2][0] = ff_put_vp8_bilin4_v_armv6;

    dsp->put_vp8_bilinear_pixels_tab[2][2][1] = ff_put_vp8_bilin4_hv_armv6;

    dsp->put_vp8_bilinear_pixels_tab[2][2][2] = ff_put_vp8_bilin4_hv_armv6;

}
