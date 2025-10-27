void ff_avg_h264_qpel16_mc02_msa(uint8_t *dst, const uint8_t *src,

                                 ptrdiff_t stride)

{

    avc_luma_vt_and_aver_dst_16x16_msa(src - (stride * 2), stride, dst, stride);

}
