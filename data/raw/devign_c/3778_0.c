static void dec_bit(DisasContext *dc)

{

    TCGv t0, t1;

    unsigned int op;

    int mem_index = cpu_mmu_index(dc->env);



    op = dc->ir & ((1 << 9) - 1);

    switch (op) {

        case 0x21:

            /* src.  */

            t0 = tcg_temp_new();



            LOG_DIS("src r%d r%d\n", dc->rd, dc->ra);

            tcg_gen_andi_tl(t0, cpu_R[dc->ra], 1);

            if (dc->rd) {

                t1 = tcg_temp_new();

                read_carry(dc, t1);

                tcg_gen_shli_tl(t1, t1, 31);



                tcg_gen_shri_tl(cpu_R[dc->rd], cpu_R[dc->ra], 1);

                tcg_gen_or_tl(cpu_R[dc->rd], cpu_R[dc->rd], t1);

                tcg_temp_free(t1);

            }



            /* Update carry.  */

            write_carry(dc, t0);

            tcg_temp_free(t0);

            break;



        case 0x1:

        case 0x41:

            /* srl.  */

            t0 = tcg_temp_new();

            LOG_DIS("srl r%d r%d\n", dc->rd, dc->ra);



            /* Update carry.  */

            tcg_gen_andi_tl(t0, cpu_R[dc->ra], 1);

            write_carry(dc, t0);

            tcg_temp_free(t0);

            if (dc->rd) {

                if (op == 0x41)

                    tcg_gen_shri_tl(cpu_R[dc->rd], cpu_R[dc->ra], 1);

                else

                    tcg_gen_sari_tl(cpu_R[dc->rd], cpu_R[dc->ra], 1);

            }

            break;

        case 0x60:

            LOG_DIS("ext8s r%d r%d\n", dc->rd, dc->ra);

            tcg_gen_ext8s_i32(cpu_R[dc->rd], cpu_R[dc->ra]);

            break;

        case 0x61:

            LOG_DIS("ext16s r%d r%d\n", dc->rd, dc->ra);

            tcg_gen_ext16s_i32(cpu_R[dc->rd], cpu_R[dc->ra]);

            break;

        case 0x64:

        case 0x66:

        case 0x74:

        case 0x76:

            /* wdc.  */

            LOG_DIS("wdc r%d\n", dc->ra);

            if ((dc->tb_flags & MSR_EE_FLAG)

                 && mem_index == MMU_USER_IDX) {

                tcg_gen_movi_tl(cpu_SR[SR_ESR], ESR_EC_PRIVINSN);

                t_gen_raise_exception(dc, EXCP_HW_EXCP);

                return;

            }

            break;

        case 0x68:

            /* wic.  */

            LOG_DIS("wic r%d\n", dc->ra);

            if ((dc->tb_flags & MSR_EE_FLAG)

                 && mem_index == MMU_USER_IDX) {

                tcg_gen_movi_tl(cpu_SR[SR_ESR], ESR_EC_PRIVINSN);

                t_gen_raise_exception(dc, EXCP_HW_EXCP);

                return;

            }

            break;

        case 0xe0:

            if ((dc->tb_flags & MSR_EE_FLAG)

                && (dc->env->pvr.regs[2] & PVR2_ILL_OPCODE_EXC_MASK)

                && !((dc->env->pvr.regs[2] & PVR2_USE_PCMP_INSTR))) {

                tcg_gen_movi_tl(cpu_SR[SR_ESR], ESR_EC_ILLEGAL_OP);

                t_gen_raise_exception(dc, EXCP_HW_EXCP);

            }

            if (dc->env->pvr.regs[2] & PVR2_USE_PCMP_INSTR) {

                gen_helper_clz(cpu_R[dc->rd], cpu_R[dc->ra]);

            }

            break;

        case 0x1e0:

            /* swapb */

            LOG_DIS("swapb r%d r%d\n", dc->rd, dc->ra);

            tcg_gen_bswap32_i32(cpu_R[dc->rd], cpu_R[dc->ra]);

            break;

        case 0x1e2:

            /*swaph */

            LOG_DIS("swaph r%d r%d\n", dc->rd, dc->ra);

            tcg_gen_rotri_i32(cpu_R[dc->rd], cpu_R[dc->ra], 16);

            break;

        default:

            cpu_abort(dc->env, "unknown bit oc=%x op=%x rd=%d ra=%d rb=%d\n",

                     dc->pc, op, dc->rd, dc->ra, dc->rb);

            break;

    }

}
