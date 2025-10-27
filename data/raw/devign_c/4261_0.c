void ff_put_h264_qpel16_mc22_msa(uint8_t *dst, const uint8_t *src,

                                 ptrdiff_t stride)

{

    avc_luma_mid_16w_msa(src - (2 * stride) - 2, stride, dst, stride, 16);

}
