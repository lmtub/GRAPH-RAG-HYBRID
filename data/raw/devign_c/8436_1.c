static void rv40_v_strong_loop_filter(uint8_t *src, const int stride,

                                      const int alpha, const int lims,

                                      const int dmode, const int chroma)

{

    rv40_strong_loop_filter(src, 1, stride, alpha, lims, dmode, chroma);

}
