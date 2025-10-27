static void dec_pattern(DisasContext *dc)

{

    unsigned int mode;

    int l1;



    if ((dc->tb_flags & MSR_EE_FLAG)

          && !(dc->env->pvr.regs[2] & PVR2_ILL_OPCODE_EXC_MASK)

          && !((dc->env->pvr.regs[2] & PVR2_USE_PCMP_INSTR))) {

        tcg_gen_movi_tl(cpu_SR[SR_ESR], ESR_EC_ILLEGAL_OP);

        t_gen_raise_exception(dc, EXCP_HW_EXCP);

    }



    mode = dc->opcode & 3;

    switch (mode) {

        case 0:

            /* pcmpbf.  */

            LOG_DIS("pcmpbf r%d r%d r%d\n", dc->rd, dc->ra, dc->rb);

            if (dc->rd)

                gen_helper_pcmpbf(cpu_R[dc->rd], cpu_R[dc->ra], cpu_R[dc->rb]);

            break;

        case 2:

            LOG_DIS("pcmpeq r%d r%d r%d\n", dc->rd, dc->ra, dc->rb);

            if (dc->rd) {

                TCGv t0 = tcg_temp_local_new();

                l1 = gen_new_label();

                tcg_gen_movi_tl(t0, 1);

                tcg_gen_brcond_tl(TCG_COND_EQ,

                                  cpu_R[dc->ra], cpu_R[dc->rb], l1);

                tcg_gen_movi_tl(t0, 0);

                gen_set_label(l1);

                tcg_gen_mov_tl(cpu_R[dc->rd], t0);

                tcg_temp_free(t0);

            }

            break;

        case 3:

            LOG_DIS("pcmpne r%d r%d r%d\n", dc->rd, dc->ra, dc->rb);

            l1 = gen_new_label();

            if (dc->rd) {

                TCGv t0 = tcg_temp_local_new();

                tcg_gen_movi_tl(t0, 1);

                tcg_gen_brcond_tl(TCG_COND_NE,

                                  cpu_R[dc->ra], cpu_R[dc->rb], l1);

                tcg_gen_movi_tl(t0, 0);

                gen_set_label(l1);

                tcg_gen_mov_tl(cpu_R[dc->rd], t0);

                tcg_temp_free(t0);

            }

            break;

        default:

            cpu_abort(dc->env,

                      "unsupported pattern insn opcode=%x\n", dc->opcode);

            break;

    }

}
