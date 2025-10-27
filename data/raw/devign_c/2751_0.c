static inline void gen_op_fcmpd(int fccno, TCGv_i64 r_rs1, TCGv_i64 r_rs2)

{

    gen_helper_fcmpd(cpu_env, r_rs1, r_rs2);

}
