void ff_rv40dsp_init_neon(RV34DSPContext *c, DSPContext* dsp)

{

    c->put_pixels_tab[0][ 1] = ff_put_rv40_qpel16_mc10_neon;

    c->put_pixels_tab[0][ 3] = ff_put_rv40_qpel16_mc30_neon;

    c->put_pixels_tab[0][ 4] = ff_put_rv40_qpel16_mc01_neon;

    c->put_pixels_tab[0][ 5] = ff_put_rv40_qpel16_mc11_neon;

    c->put_pixels_tab[0][ 6] = ff_put_rv40_qpel16_mc21_neon;

    c->put_pixels_tab[0][ 7] = ff_put_rv40_qpel16_mc31_neon;

    c->put_pixels_tab[0][ 9] = ff_put_rv40_qpel16_mc12_neon;

    c->put_pixels_tab[0][10] = ff_put_rv40_qpel16_mc22_neon;

    c->put_pixels_tab[0][11] = ff_put_rv40_qpel16_mc32_neon;

    c->put_pixels_tab[0][12] = ff_put_rv40_qpel16_mc03_neon;

    c->put_pixels_tab[0][13] = ff_put_rv40_qpel16_mc13_neon;

    c->put_pixels_tab[0][14] = ff_put_rv40_qpel16_mc23_neon;

    c->put_pixels_tab[0][15] = ff_put_rv40_qpel16_mc33_neon;

    c->avg_pixels_tab[0][ 1] = ff_avg_rv40_qpel16_mc10_neon;

    c->avg_pixels_tab[0][ 3] = ff_avg_rv40_qpel16_mc30_neon;

    c->avg_pixels_tab[0][ 4] = ff_avg_rv40_qpel16_mc01_neon;

    c->avg_pixels_tab[0][ 5] = ff_avg_rv40_qpel16_mc11_neon;

    c->avg_pixels_tab[0][ 6] = ff_avg_rv40_qpel16_mc21_neon;

    c->avg_pixels_tab[0][ 7] = ff_avg_rv40_qpel16_mc31_neon;

    c->avg_pixels_tab[0][ 9] = ff_avg_rv40_qpel16_mc12_neon;

    c->avg_pixels_tab[0][10] = ff_avg_rv40_qpel16_mc22_neon;

    c->avg_pixels_tab[0][11] = ff_avg_rv40_qpel16_mc32_neon;

    c->avg_pixels_tab[0][12] = ff_avg_rv40_qpel16_mc03_neon;

    c->avg_pixels_tab[0][13] = ff_avg_rv40_qpel16_mc13_neon;

    c->avg_pixels_tab[0][14] = ff_avg_rv40_qpel16_mc23_neon;

    c->avg_pixels_tab[0][15] = ff_avg_rv40_qpel16_mc33_neon;

    c->put_pixels_tab[1][ 1] = ff_put_rv40_qpel8_mc10_neon;

    c->put_pixels_tab[1][ 3] = ff_put_rv40_qpel8_mc30_neon;

    c->put_pixels_tab[1][ 4] = ff_put_rv40_qpel8_mc01_neon;

    c->put_pixels_tab[1][ 5] = ff_put_rv40_qpel8_mc11_neon;

    c->put_pixels_tab[1][ 6] = ff_put_rv40_qpel8_mc21_neon;

    c->put_pixels_tab[1][ 7] = ff_put_rv40_qpel8_mc31_neon;

    c->put_pixels_tab[1][ 9] = ff_put_rv40_qpel8_mc12_neon;

    c->put_pixels_tab[1][10] = ff_put_rv40_qpel8_mc22_neon;

    c->put_pixels_tab[1][11] = ff_put_rv40_qpel8_mc32_neon;

    c->put_pixels_tab[1][12] = ff_put_rv40_qpel8_mc03_neon;

    c->put_pixels_tab[1][13] = ff_put_rv40_qpel8_mc13_neon;

    c->put_pixels_tab[1][14] = ff_put_rv40_qpel8_mc23_neon;

    c->put_pixels_tab[1][15] = ff_put_rv40_qpel8_mc33_neon;

    c->avg_pixels_tab[1][ 1] = ff_avg_rv40_qpel8_mc10_neon;

    c->avg_pixels_tab[1][ 3] = ff_avg_rv40_qpel8_mc30_neon;

    c->avg_pixels_tab[1][ 4] = ff_avg_rv40_qpel8_mc01_neon;

    c->avg_pixels_tab[1][ 5] = ff_avg_rv40_qpel8_mc11_neon;

    c->avg_pixels_tab[1][ 6] = ff_avg_rv40_qpel8_mc21_neon;

    c->avg_pixels_tab[1][ 7] = ff_avg_rv40_qpel8_mc31_neon;

    c->avg_pixels_tab[1][ 9] = ff_avg_rv40_qpel8_mc12_neon;

    c->avg_pixels_tab[1][10] = ff_avg_rv40_qpel8_mc22_neon;

    c->avg_pixels_tab[1][11] = ff_avg_rv40_qpel8_mc32_neon;

    c->avg_pixels_tab[1][12] = ff_avg_rv40_qpel8_mc03_neon;

    c->avg_pixels_tab[1][13] = ff_avg_rv40_qpel8_mc13_neon;

    c->avg_pixels_tab[1][14] = ff_avg_rv40_qpel8_mc23_neon;

    c->avg_pixels_tab[1][15] = ff_avg_rv40_qpel8_mc33_neon;



    c->put_chroma_pixels_tab[0] = ff_put_rv40_chroma_mc8_neon;

    c->put_chroma_pixels_tab[1] = ff_put_rv40_chroma_mc4_neon;

    c->avg_chroma_pixels_tab[0] = ff_avg_rv40_chroma_mc8_neon;

    c->avg_chroma_pixels_tab[1] = ff_avg_rv40_chroma_mc4_neon;



    c->rv40_weight_pixels_tab[0][0] = ff_rv40_weight_func_16_neon;

    c->rv40_weight_pixels_tab[0][1] = ff_rv40_weight_func_8_neon;



    c->rv40_loop_filter_strength[0] = ff_rv40_h_loop_filter_strength_neon;

    c->rv40_loop_filter_strength[1] = ff_rv40_v_loop_filter_strength_neon;

    c->rv40_weak_loop_filter[0]     = ff_rv40_h_weak_loop_filter_neon;

    c->rv40_weak_loop_filter[1]     = ff_rv40_v_weak_loop_filter_neon;

}
