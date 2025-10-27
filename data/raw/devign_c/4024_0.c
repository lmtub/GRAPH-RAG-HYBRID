static void avc_wgt_8width_msa(uint8_t *data,

                               int32_t stride,

                               int32_t height,

                               int32_t log2_denom,

                               int32_t src_weight,

                               int32_t offset_in)

{

    uint8_t cnt;

    v16u8 zero = { 0 };

    v16u8 src0, src1, src2, src3;

    v8u16 src0_r, src1_r, src2_r, src3_r;

    v8u16 temp0, temp1, temp2, temp3;

    v8u16 wgt, denom, offset;



    offset_in <<= (log2_denom);



    if (log2_denom) {

        offset_in += (1 << (log2_denom - 1));

    }



    wgt = (v8u16) __msa_fill_h(src_weight);

    offset = (v8u16) __msa_fill_h(offset_in);

    denom = (v8u16) __msa_fill_h(log2_denom);



    for (cnt = height / 4; cnt--;) {

        LOAD_4VECS_UB(data, stride, src0, src1, src2, src3);



        ILVR_B_4VECS_UH(src0, src1, src2, src3, zero, zero, zero, zero,

                        src0_r, src1_r, src2_r, src3_r);



        temp0 = wgt * src0_r;

        temp1 = wgt * src1_r;

        temp2 = wgt * src2_r;

        temp3 = wgt * src3_r;



        ADDS_S_H_4VECS_UH(temp0, offset, temp1, offset,

                          temp2, offset, temp3, offset,

                          temp0, temp1, temp2, temp3);



        MAXI_S_H_4VECS_UH(temp0, temp1, temp2, temp3, 0);



        SRL_H_4VECS_UH(temp0, temp1, temp2, temp3,

                       temp0, temp1, temp2, temp3, denom);



        SAT_U_H_4VECS_UH(temp0, temp1, temp2, temp3, 7);



        PCKEV_B_STORE_8_BYTES_4(temp0, temp1, temp2, temp3, data, stride);

        data += (4 * stride);

    }

}
