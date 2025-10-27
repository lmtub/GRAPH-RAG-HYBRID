static av_noinline void emulated_edge_mc_sse(uint8_t * buf,const uint8_t *src,

                                             ptrdiff_t buf_stride,

                                             ptrdiff_t src_stride,

                                             int block_w, int block_h,

                                             int src_x, int src_y, int w, int h)

{

    emulated_edge_mc(buf, src, buf_stride, src_stride, block_w, block_h,

                     src_x, src_y, w, h, vfixtbl_sse, &ff_emu_edge_vvar_sse,

                     hfixtbl_sse, &ff_emu_edge_hvar_sse);

}
