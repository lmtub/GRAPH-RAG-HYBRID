static inline void gen_outs(DisasContext *s, TCGMemOp ot)

{

    if (use_icount)

        gen_io_start();

    gen_string_movl_A0_ESI(s);

    gen_op_ld_v(s, ot, cpu_T[0], cpu_A0);



    tcg_gen_trunc_tl_i32(cpu_tmp2_i32, cpu_regs[R_EDX]);

    tcg_gen_andi_i32(cpu_tmp2_i32, cpu_tmp2_i32, 0xffff);

    tcg_gen_trunc_tl_i32(cpu_tmp3_i32, cpu_T[0]);

    gen_helper_out_func(ot, cpu_tmp2_i32, cpu_tmp3_i32);



    gen_op_movl_T0_Dshift(ot);

    gen_op_add_reg_T0(s->aflag, R_ESI);

    if (use_icount)

        gen_io_end();

}
