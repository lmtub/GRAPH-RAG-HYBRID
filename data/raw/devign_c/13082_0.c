void ff_avg_h264_qpel4_mc22_msa(uint8_t *dst, const uint8_t *src,

                                ptrdiff_t stride)

{

    avc_luma_mid_and_aver_dst_4x4_msa(src - (2 * stride) - 2,

                                      stride, dst, stride);

}
