static void avc_wgt_16width_msa(uint8_t *data,

                                int32_t stride,

                                int32_t height,

                                int32_t log2_denom,

                                int32_t src_weight,

                                int32_t offset_in)

{

    uint8_t cnt;

    v16u8 zero = { 0 };

    v16u8 src0, src1, src2, src3;

    v16u8 dst0, dst1, dst2, dst3;

    v8u16 src0_l, src1_l, src2_l, src3_l;

    v8u16 src0_r, src1_r, src2_r, src3_r;

    v8u16 temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7;

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



        ILV_B_LRLR_UH(src0, zero, src1, zero, src0_l, src0_r, src1_l, src1_r);

        ILV_B_LRLR_UH(src2, zero, src3, zero, src2_l, src2_r, src3_l, src3_r);



        temp0 = wgt * src0_r;

        temp1 = wgt * src0_l;

        temp2 = wgt * src1_r;

        temp3 = wgt * src1_l;

        temp4 = wgt * src2_r;

        temp5 = wgt * src2_l;

        temp6 = wgt * src3_r;

        temp7 = wgt * src3_l;



        ADDS_S_H_4VECS_UH(temp0, offset, temp1, offset,

                          temp2, offset, temp3, offset,

                          temp0, temp1, temp2, temp3);



        ADDS_S_H_4VECS_UH(temp4, offset, temp5, offset,

                          temp6, offset, temp7, offset,

                          temp4, temp5, temp6, temp7);



        MAXI_S_H_4VECS_UH(temp0, temp1, temp2, temp3, 0);

        MAXI_S_H_4VECS_UH(temp4, temp5, temp6, temp7, 0);



        SRL_H_4VECS_UH(temp0, temp1, temp2, temp3,

                       temp0, temp1, temp2, temp3, denom);



        SRL_H_4VECS_UH(temp4, temp5, temp6, temp7,

                       temp4, temp5, temp6, temp7, denom);



        SAT_U_H_4VECS_UH(temp0, temp1, temp2, temp3, 7);

        SAT_U_H_4VECS_UH(temp4, temp5, temp6, temp7, 7);



        PCKEV_B_4VECS_UB(temp1, temp3, temp5, temp7, temp0, temp2, temp4, temp6,

                         dst0, dst1, dst2, dst3);



        STORE_4VECS_UB(data, stride, dst0, dst1, dst2, dst3);

        data += 4 * stride;

    }

}
