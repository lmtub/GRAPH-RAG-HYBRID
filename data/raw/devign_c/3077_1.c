static av_noinline void emulated_edge_mc_sse(uint8_t *buf, ptrdiff_t buf_stride,

                                             const uint8_t *src, ptrdiff_t src_stride,

                                             int block_w, int block_h,

                                             int src_x, int src_y, int w, int h)

{

    emulated_edge_mc(buf, buf_stride, src, src_stride, block_w, block_h, src_x,

                     src_y, w, h, vfixtbl_sse, &ff_emu_edge_vvar_sse, hfixtbl_sse,

#if ARCH_X86_64

                     &ff_emu_edge_hvar_sse

#else

                     &ff_emu_edge_hvar_mmx

#endif

                     );

}
