void ff_weight_h264_pixels4_8_msa(uint8_t *src, int stride,

                                  int height, int log2_denom,

                                  int weight_src, int offset)

{

    avc_wgt_4width_msa(src, stride,

                       height, log2_denom, weight_src, offset);

}
