static void intra_predict_vert_8x8_msa(uint8_t *src, uint8_t *dst,

                                       int32_t dst_stride)

{

    uint32_t row;

    uint32_t src_data1, src_data2;



    src_data1 = LW(src);

    src_data2 = LW(src + 4);



    for (row = 8; row--;) {

        SW(src_data1, dst);

        SW(src_data2, (dst + 4));

        dst += dst_stride;

    }

}
