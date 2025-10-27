static inline void gen_intermediate_code_internal(OpenRISCCPU *cpu,

                                                  TranslationBlock *tb,

                                                  int search_pc)

{

    CPUState *cs = CPU(cpu);

    struct DisasContext ctx, *dc = &ctx;

    uint16_t *gen_opc_end;

    uint32_t pc_start;

    int j, k;

    uint32_t next_page_start;

    int num_insns;

    int max_insns;



    pc_start = tb->pc;

    dc->tb = tb;



    gen_opc_end = tcg_ctx.gen_opc_buf + OPC_MAX_SIZE;

    dc->is_jmp = DISAS_NEXT;

    dc->ppc = pc_start;

    dc->pc = pc_start;

    dc->flags = cpu->env.cpucfgr;

    dc->mem_idx = cpu_mmu_index(&cpu->env);

    dc->synced_flags = dc->tb_flags = tb->flags;

    dc->delayed_branch = !!(dc->tb_flags & D_FLAG);

    dc->singlestep_enabled = cs->singlestep_enabled;

    if (qemu_loglevel_mask(CPU_LOG_TB_IN_ASM)) {

        qemu_log("-----------------------------------------\n");

        log_cpu_state(CPU(cpu), 0);

    }



    next_page_start = (pc_start & TARGET_PAGE_MASK) + TARGET_PAGE_SIZE;

    k = -1;

    num_insns = 0;

    max_insns = tb->cflags & CF_COUNT_MASK;



    if (max_insns == 0) {

        max_insns = CF_COUNT_MASK;

    }



    gen_tb_start();



    do {

        check_breakpoint(cpu, dc);

        if (search_pc) {

            j = tcg_ctx.gen_opc_ptr - tcg_ctx.gen_opc_buf;

            if (k < j) {

                k++;

                while (k < j) {

                    tcg_ctx.gen_opc_instr_start[k++] = 0;

                }

            }

            tcg_ctx.gen_opc_pc[k] = dc->pc;

            tcg_ctx.gen_opc_instr_start[k] = 1;

            tcg_ctx.gen_opc_icount[k] = num_insns;

        }



        if (unlikely(qemu_loglevel_mask(CPU_LOG_TB_OP | CPU_LOG_TB_OP_OPT))) {

            tcg_gen_debug_insn_start(dc->pc);

        }



        if (num_insns + 1 == max_insns && (tb->cflags & CF_LAST_IO)) {

            gen_io_start();

        }

        dc->ppc = dc->pc - 4;

        dc->npc = dc->pc + 4;

        tcg_gen_movi_tl(cpu_ppc, dc->ppc);

        tcg_gen_movi_tl(cpu_npc, dc->npc);

        disas_openrisc_insn(dc, cpu);

        dc->pc = dc->npc;

        num_insns++;

        /* delay slot */

        if (dc->delayed_branch) {

            dc->delayed_branch--;

            if (!dc->delayed_branch) {

                dc->tb_flags &= ~D_FLAG;

                gen_sync_flags(dc);

                tcg_gen_mov_tl(cpu_pc, jmp_pc);

                tcg_gen_mov_tl(cpu_npc, jmp_pc);

                tcg_gen_movi_tl(jmp_pc, 0);

                tcg_gen_exit_tb(0);

                dc->is_jmp = DISAS_JUMP;

                break;

            }

        }

    } while (!dc->is_jmp

             && tcg_ctx.gen_opc_ptr < gen_opc_end

             && !cs->singlestep_enabled

             && !singlestep

             && (dc->pc < next_page_start)

             && num_insns < max_insns);



    if (tb->cflags & CF_LAST_IO) {

        gen_io_end();

    }

    if (dc->is_jmp == DISAS_NEXT) {

        dc->is_jmp = DISAS_UPDATE;

        tcg_gen_movi_tl(cpu_pc, dc->pc);

    }

    if (unlikely(cs->singlestep_enabled)) {

        if (dc->is_jmp == DISAS_NEXT) {

            tcg_gen_movi_tl(cpu_pc, dc->pc);

        }

        gen_exception(dc, EXCP_DEBUG);

    } else {

        switch (dc->is_jmp) {

        case DISAS_NEXT:

            gen_goto_tb(dc, 0, dc->pc);

            break;

        default:

        case DISAS_JUMP:

            break;

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

        k++;

        while (k <= j) {

            tcg_ctx.gen_opc_instr_start[k++] = 0;

        }

    } else {

        tb->size = dc->pc - pc_start;

        tb->icount = num_insns;

    }



#ifdef DEBUG_DISAS

    if (qemu_loglevel_mask(CPU_LOG_TB_IN_ASM)) {

        qemu_log("\n");

        log_target_disas(&cpu->env, pc_start, dc->pc - pc_start, 0);

        qemu_log("\nisize=%d osize=%td\n",

            dc->pc - pc_start, tcg_ctx.gen_opc_ptr -

            tcg_ctx.gen_opc_buf);

    }

#endif

}
