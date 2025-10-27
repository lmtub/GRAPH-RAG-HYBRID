void ff_put_h264_qpel8_mc00_msa(uint8_t *dst, const uint8_t *src,

                                ptrdiff_t stride)

{

    copy_width8_msa(src, stride, dst, stride, 8);

}
