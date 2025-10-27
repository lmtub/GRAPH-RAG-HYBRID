static void intra_predict_vert_16x16_msa(uint8_t *src, uint8_t *dst,

                                         int32_t dst_stride)

{

    uint32_t row;

    v16u8 src0;



    src0 = LD_UB(src);



    for (row = 16; row--;) {

        ST_UB(src0, dst);

        dst += dst_stride;

    }

}
