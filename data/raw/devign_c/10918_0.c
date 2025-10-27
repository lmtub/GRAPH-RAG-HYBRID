static void avc_luma_hv_qrt_16w_msa(const uint8_t *src_x, const uint8_t *src_y,

                                    int32_t src_stride, uint8_t *dst,

                                    int32_t dst_stride, int32_t height)

{

    uint32_t multiple8_cnt;



    for (multiple8_cnt = 2; multiple8_cnt--;) {

        avc_luma_hv_qrt_8w_msa(src_x, src_y, src_stride, dst, dst_stride,

                               height);



        src_x += 8;

        src_y += 8;

        dst += 8;

    }

}
