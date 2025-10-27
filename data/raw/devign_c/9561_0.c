static void dec_mul(DisasContext *dc)

{

    TCGv d[2];

    unsigned int subcode;



    if ((dc->tb_flags & MSR_EE_FLAG)

         && !(dc->env->pvr.regs[2] & PVR2_ILL_OPCODE_EXC_MASK)

         && !(dc->env->pvr.regs[0] & PVR0_USE_HW_MUL_MASK)) {

        tcg_gen_movi_tl(cpu_SR[SR_ESR], ESR_EC_ILLEGAL_OP);

        t_gen_raise_exception(dc, EXCP_HW_EXCP);

        return;

    }



    subcode = dc->imm & 3;

    d[0] = tcg_temp_new();

    d[1] = tcg_temp_new();



    if (dc->type_b) {

        LOG_DIS("muli r%d r%d %x\n", dc->rd, dc->ra, dc->imm);

        t_gen_mulu(cpu_R[dc->rd], d[1], cpu_R[dc->ra], *(dec_alu_op_b(dc)));

        goto done;

    }



    /* mulh, mulhsu and mulhu are not available if C_USE_HW_MUL is < 2.  */

    if (subcode >= 1 && subcode <= 3

        && !((dc->env->pvr.regs[2] & PVR2_USE_MUL64_MASK))) {

        /* nop??? */

    }



    switch (subcode) {

        case 0:

            LOG_DIS("mul r%d r%d r%d\n", dc->rd, dc->ra, dc->rb);

            t_gen_mulu(cpu_R[dc->rd], d[1], cpu_R[dc->ra], cpu_R[dc->rb]);

            break;

        case 1:

            LOG_DIS("mulh r%d r%d r%d\n", dc->rd, dc->ra, dc->rb);

            t_gen_muls(d[0], cpu_R[dc->rd], cpu_R[dc->ra], cpu_R[dc->rb]);

            break;

        case 2:

            LOG_DIS("mulhsu r%d r%d r%d\n", dc->rd, dc->ra, dc->rb);

            t_gen_muls(d[0], cpu_R[dc->rd], cpu_R[dc->ra], cpu_R[dc->rb]);

            break;

        case 3:

            LOG_DIS("mulhu r%d r%d r%d\n", dc->rd, dc->ra, dc->rb);

            t_gen_mulu(d[0], cpu_R[dc->rd], cpu_R[dc->ra], cpu_R[dc->rb]);

            break;

        default:

            cpu_abort(dc->env, "unknown MUL insn %x\n", subcode);

            break;

    }

done:

    tcg_temp_free(d[0]);

    tcg_temp_free(d[1]);

}
