static void h264_h_loop_filter_luma_intra_c(uint8_t *pix, int stride, int alpha, int beta)

{

    h264_loop_filter_luma_intra_c(pix, 1, stride, alpha, beta);

}
