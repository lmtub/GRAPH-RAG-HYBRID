static inline void gen_ins(DisasContext *s, TCGMemOp ot)

{

    if (use_icount)

        gen_io_start();

    gen_string_movl_A0_EDI(s);

    /* Note: we must do this dummy write first to be restartable in

       case of page fault. */

    tcg_gen_movi_tl(cpu_T[0], 0);

    gen_op_st_v(s, ot, cpu_T[0], cpu_A0);

    tcg_gen_trunc_tl_i32(cpu_tmp2_i32, cpu_regs[R_EDX]);

    tcg_gen_andi_i32(cpu_tmp2_i32, cpu_tmp2_i32, 0xffff);

    gen_helper_in_func(ot, cpu_T[0], cpu_tmp2_i32);

    gen_op_st_v(s, ot, cpu_T[0], cpu_A0);

    gen_op_movl_T0_Dshift(ot);

    gen_op_add_reg_T0(s->aflag, R_EDI);

    if (use_icount)

        gen_io_end();

}
