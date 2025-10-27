static inline void gen_op_multiply(TCGv dst, TCGv src1, TCGv src2, int sign_ext)

{

    TCGv_i32 r_src1, r_src2;

    TCGv_i64 r_temp, r_temp2;



    r_src1 = tcg_temp_new_i32();

    r_src2 = tcg_temp_new_i32();



    tcg_gen_trunc_tl_i32(r_src1, src1);

    tcg_gen_trunc_tl_i32(r_src2, src2);



    r_temp = tcg_temp_new_i64();

    r_temp2 = tcg_temp_new_i64();



    if (sign_ext) {

        tcg_gen_ext_i32_i64(r_temp, r_src2);

        tcg_gen_ext_i32_i64(r_temp2, r_src1);

    } else {

        tcg_gen_extu_i32_i64(r_temp, r_src2);

        tcg_gen_extu_i32_i64(r_temp2, r_src1);

    }



    tcg_gen_mul_i64(r_temp2, r_temp, r_temp2);



    tcg_gen_shri_i64(r_temp, r_temp2, 32);

    tcg_gen_trunc_i64_tl(cpu_tmp0, r_temp);

    tcg_temp_free_i64(r_temp);

    tcg_gen_andi_tl(cpu_y, cpu_tmp0, 0xffffffff);



    tcg_gen_trunc_i64_tl(dst, r_temp2);



    tcg_temp_free_i64(r_temp2);



    tcg_temp_free_i32(r_src1);

    tcg_temp_free_i32(r_src2);

}
