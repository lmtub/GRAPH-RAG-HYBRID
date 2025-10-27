static void avc_biwgt_16width_msa(uint8_t *src, int32_t src_stride,

                                  uint8_t *dst, int32_t dst_stride,

                                  int32_t height, int32_t log2_denom,

                                  int32_t src_weight, int32_t dst_weight,

                                  int32_t offset_in)

{

    uint8_t cnt;

    v16i8 src_wgt, dst_wgt, wgt;

    v16i8 src0, src1, src2, src3;

    v16i8 dst0, dst1, dst2, dst3;

    v16i8 vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7;

    v8i16 temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7;

    v8i16 denom, offset, add_val;

    int32_t val = 128 * (src_weight + dst_weight);



    offset_in = ((offset_in + 1) | 1) << log2_denom;



    src_wgt = __msa_fill_b(src_weight);

    dst_wgt = __msa_fill_b(dst_weight);

    offset = __msa_fill_h(offset_in);

    denom = __msa_fill_h(log2_denom + 1);

    add_val = __msa_fill_h(val);

    offset += add_val;



    wgt = __msa_ilvev_b(dst_wgt, src_wgt);



    for (cnt = height / 4; cnt--;) {

        LOAD_4VECS_SB(src, src_stride, src0, src1, src2, src3);

        src += (4 * src_stride);



        LOAD_4VECS_SB(dst, dst_stride, dst0, dst1, dst2, dst3);



        XORI_B_4VECS_SB(src0, src1, src2, src3, src0, src1, src2, src3, 128);



        XORI_B_4VECS_SB(dst0, dst1, dst2, dst3, dst0, dst1, dst2, dst3, 128);



        ILV_B_LRLR_SB(src0, dst0, src1, dst1, vec1, vec0, vec3, vec2);

        ILV_B_LRLR_SB(src2, dst2, src3, dst3, vec5, vec4, vec7, vec6);



        temp0 = __msa_dpadd_s_h(offset, wgt, vec0);

        temp1 = __msa_dpadd_s_h(offset, wgt, vec1);

        temp2 = __msa_dpadd_s_h(offset, wgt, vec2);

        temp3 = __msa_dpadd_s_h(offset, wgt, vec3);

        temp4 = __msa_dpadd_s_h(offset, wgt, vec4);

        temp5 = __msa_dpadd_s_h(offset, wgt, vec5);

        temp6 = __msa_dpadd_s_h(offset, wgt, vec6);

        temp7 = __msa_dpadd_s_h(offset, wgt, vec7);



        SRA_4VECS(temp0, temp1, temp2, temp3,

                  temp0, temp1, temp2, temp3, denom);

        SRA_4VECS(temp4, temp5, temp6, temp7,

                  temp4, temp5, temp6, temp7, denom);



        temp0 = CLIP_UNSIGNED_CHAR_H(temp0);

        temp1 = CLIP_UNSIGNED_CHAR_H(temp1);

        temp2 = CLIP_UNSIGNED_CHAR_H(temp2);

        temp3 = CLIP_UNSIGNED_CHAR_H(temp3);

        temp4 = CLIP_UNSIGNED_CHAR_H(temp4);

        temp5 = CLIP_UNSIGNED_CHAR_H(temp5);

        temp6 = CLIP_UNSIGNED_CHAR_H(temp6);

        temp7 = CLIP_UNSIGNED_CHAR_H(temp7);



        PCKEV_B_4VECS_SB(temp1, temp3, temp5, temp7, temp0, temp2, temp4, temp6,

                         dst0, dst1, dst2, dst3);



        STORE_4VECS_SB(dst, dst_stride, dst0, dst1, dst2, dst3);

        dst += 4 * dst_stride;

    }

}
