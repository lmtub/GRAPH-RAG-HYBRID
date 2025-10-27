void ff_put_h264_qpel16_mc11_msa(uint8_t *dst, const uint8_t *src,

                                 ptrdiff_t stride)

{

    avc_luma_hv_qrt_16w_msa(src - 2,

                            src - (stride * 2), stride, dst, stride, 16);

}
