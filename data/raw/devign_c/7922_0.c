static void avc_luma_mid_16w_msa(const uint8_t *src, int32_t src_stride,

                                 uint8_t *dst, int32_t dst_stride,

                                 int32_t height)

{

    uint32_t multiple8_cnt;



    for (multiple8_cnt = 2; multiple8_cnt--;) {

        avc_luma_mid_8w_msa(src, src_stride, dst, dst_stride, height);

        src += 8;

        dst += 8;

    }

}
