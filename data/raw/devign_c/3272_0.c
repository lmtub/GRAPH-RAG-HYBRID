static void dec_barrel(DisasContext *dc)

{

    TCGv t0;

    unsigned int s, t;



    if ((dc->tb_flags & MSR_EE_FLAG)

          && !(dc->env->pvr.regs[2] & PVR2_ILL_OPCODE_EXC_MASK)

          && !(dc->env->pvr.regs[0] & PVR0_USE_BARREL_MASK)) {

        tcg_gen_movi_tl(cpu_SR[SR_ESR], ESR_EC_ILLEGAL_OP);

        t_gen_raise_exception(dc, EXCP_HW_EXCP);

        return;

    }



    s = dc->imm & (1 << 10);

    t = dc->imm & (1 << 9);



    LOG_DIS("bs%s%s r%d r%d r%d\n",

            s ? "l" : "r", t ? "a" : "l", dc->rd, dc->ra, dc->rb);



    t0 = tcg_temp_new();



    tcg_gen_mov_tl(t0, *(dec_alu_op_b(dc)));

    tcg_gen_andi_tl(t0, t0, 31);



    if (s)

        tcg_gen_shl_tl(cpu_R[dc->rd], cpu_R[dc->ra], t0);

    else {

        if (t)

            tcg_gen_sar_tl(cpu_R[dc->rd], cpu_R[dc->ra], t0);

        else

            tcg_gen_shr_tl(cpu_R[dc->rd], cpu_R[dc->ra], t0);

    }

}
