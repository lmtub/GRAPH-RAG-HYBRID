static void h264_h_loop_filter_chroma_c(uint8_t *pix, int stride, int alpha, int beta, int8_t *tc0)

{

    h264_loop_filter_chroma_c(pix, 1, stride, alpha, beta, tc0);

}
