void gen_intermediate_code_internal(LM32CPU *cpu,

                                    TranslationBlock *tb, bool search_pc)

{

    CPUState *cs = CPU(cpu);

    CPULM32State *env = &cpu->env;

    struct DisasContext ctx, *dc = &ctx;

    uint16_t *gen_opc_end;

    uint32_t pc_start;

    int j, lj;

    uint32_t next_page_start;

    int num_insns;

    int max_insns;



    pc_start = tb->pc;

    dc->env = env;

    dc->tb = tb;



    gen_opc_end = tcg_ctx.gen_opc_buf + OPC_MAX_SIZE;



    dc->is_jmp = DISAS_NEXT;

    dc->pc = pc_start;

    dc->singlestep_enabled = cs->singlestep_enabled;

    dc->nr_nops = 0;



    if (pc_start & 3) {

        cpu_abort(env, "LM32: unaligned PC=%x\n", pc_start);

    }



    next_page_start = (pc_start & TARGET_PAGE_MASK) + TARGET_PAGE_SIZE;

    lj = -1;

    num_insns = 0;

    max_insns = tb->cflags & CF_COUNT_MASK;

    if (max_insns == 0) {

        max_insns = CF_COUNT_MASK;

    }



    gen_tb_start();

    do {

        check_breakpoint(env, dc);



        if (search_pc) {

            j = tcg_ctx.gen_opc_ptr - tcg_ctx.gen_opc_buf;

            if (lj < j) {

                lj++;

                while (lj < j) {

                    tcg_ctx.gen_opc_instr_start[lj++] = 0;

                }

            }

            tcg_ctx.gen_opc_pc[lj] = dc->pc;

            tcg_ctx.gen_opc_instr_start[lj] = 1;

            tcg_ctx.gen_opc_icount[lj] = num_insns;

        }



        /* Pretty disas.  */

        LOG_DIS("%8.8x:\t", dc->pc);



        if (num_insns + 1 == max_insns && (tb->cflags & CF_LAST_IO)) {

            gen_io_start();

        }



        decode(dc, cpu_ldl_code(env, dc->pc));

        dc->pc += 4;

        num_insns++;



    } while (!dc->is_jmp

         && tcg_ctx.gen_opc_ptr < gen_opc_end

         && !cs->singlestep_enabled

         && !singlestep

         && (dc->pc < next_page_start)

         && num_insns < max_insns);



    if (tb->cflags & CF_LAST_IO) {

        gen_io_end();

    }



    if (unlikely(cs->singlestep_enabled)) {

        if (dc->is_jmp == DISAS_NEXT) {

            tcg_gen_movi_tl(cpu_pc, dc->pc);

        }

        t_gen_raise_exception(dc, EXCP_DEBUG);

    } else {

        switch (dc->is_jmp) {

        case DISAS_NEXT:

            gen_goto_tb(dc, 1, dc->pc);

            break;

        default:

        case DISAS_JUMP:

        case DISAS_UPDATE:

            /* indicate that the hash table must be used

               to find the next TB */

            tcg_gen_exit_tb(0);

            break;

        case DISAS_TB_JUMP:

            /* nothing more to generate */

            break;

        }

    }



    gen_tb_end(tb, num_insns);

    *tcg_ctx.gen_opc_ptr = INDEX_op_end;

    if (search_pc) {

        j = tcg_ctx.gen_opc_ptr - tcg_ctx.gen_opc_buf;

        lj++;

        while (lj <= j) {

            tcg_ctx.gen_opc_instr_start[lj++] = 0;

        }

    } else {

        tb->size = dc->pc - pc_start;

        tb->icount = num_insns;

    }



#ifdef DEBUG_DISAS

    if (qemu_loglevel_mask(CPU_LOG_TB_IN_ASM)) {

        qemu_log("\n");

        log_target_disas(env, pc_start, dc->pc - pc_start, 0);

        qemu_log("\nisize=%d osize=%td\n",

            dc->pc - pc_start, tcg_ctx.gen_opc_ptr -

            tcg_ctx.gen_opc_buf);

    }

#endif

}
