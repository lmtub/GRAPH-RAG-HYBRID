static inline int gen_intermediate_code_internal(TranslationBlock * tb,

						 int spc, CPUSPARCState *env)

{

    target_ulong pc_start, last_pc;

    uint16_t *gen_opc_end;

    DisasContext dc1, *dc = &dc1;

    int j, lj = -1;



    memset(dc, 0, sizeof(DisasContext));

    dc->tb = tb;

    pc_start = tb->pc;

    dc->pc = pc_start;

    dc->npc = (target_ulong) tb->cs_base;

#if defined(CONFIG_USER_ONLY)

    dc->mem_idx = 0;

#else

    dc->mem_idx = ((env->psrs) != 0);

#endif

    gen_opc_ptr = gen_opc_buf;

    gen_opc_end = gen_opc_buf + OPC_MAX_SIZE;

    gen_opparam_ptr = gen_opparam_buf;



    env->access_type = ACCESS_CODE;



    do {

        if (env->nb_breakpoints > 0) {

            for(j = 0; j < env->nb_breakpoints; j++) {

                if (env->breakpoints[j] == dc->pc) {

                    gen_debug(dc, dc->pc);

                    break;

                }

            }

        }

        if (spc) {

            if (loglevel > 0)

                fprintf(logfile, "Search PC...\n");

            j = gen_opc_ptr - gen_opc_buf;

            if (lj < j) {

                lj++;

                while (lj < j)

                    gen_opc_instr_start[lj++] = 0;

                gen_opc_pc[lj] = dc->pc;

                gen_opc_npc[lj] = dc->npc;

                gen_opc_instr_start[lj] = 1;

            }

        }

	last_pc = dc->pc;

	disas_sparc_insn(dc);

	if (dc->is_br)

	    break;

	/* if the next PC is different, we abort now */

	if (dc->pc != (last_pc + 4))

	    break;

    } while ((gen_opc_ptr < gen_opc_end) &&

	     (dc->pc - pc_start) < (TARGET_PAGE_SIZE - 32));

    if (!dc->is_br) {

        if (dc->pc != DYNAMIC_PC && 

            (dc->npc != DYNAMIC_PC && dc->npc != JUMP_PC)) {

            /* static PC and NPC: we can use direct chaining */

            gen_op_branch((long)tb, dc->pc, dc->npc);

        } else {

            if (dc->pc != DYNAMIC_PC)

                gen_op_jmp_im(dc->pc);

            save_npc(dc);

            gen_op_movl_T0_0();

            gen_op_exit_tb();

        }

    }

    *gen_opc_ptr = INDEX_op_end;

    if (spc) {

        j = gen_opc_ptr - gen_opc_buf;

        lj++;

        while (lj <= j)

            gen_opc_instr_start[lj++] = 0;

        tb->size = 0;

#if 0

        if (loglevel > 0) {

            page_dump(logfile);

        }

#endif

    } else {

        tb->size = dc->npc - pc_start;

    }

#ifdef DEBUG_DISAS

    if (loglevel & CPU_LOG_TB_IN_ASM) {

	fprintf(logfile, "--------------\n");

	fprintf(logfile, "IN: %s\n", lookup_symbol((uint8_t *)pc_start));

	disas(logfile, (uint8_t *)pc_start, last_pc + 4 - pc_start, 0, 0);

	fprintf(logfile, "\n");

        if (loglevel & CPU_LOG_TB_OP) {

            fprintf(logfile, "OP:\n");

            dump_ops(gen_opc_buf, gen_opparam_buf);

            fprintf(logfile, "\n");

        }

    }

#endif



    env->access_type = ACCESS_DATA;

    return 0;

}
