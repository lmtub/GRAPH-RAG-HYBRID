static inline int gen_intermediate_code_internal(CPUState *env, 
                                                 TranslationBlock *tb, 
                                                 int search_pc)
{
    DisasContext dc1, *dc = &dc1;
    uint16_t *gen_opc_end;
    int j, lj;
    target_ulong pc_start;
    uint32_t next_page_start;
    /* generate intermediate code */
    pc_start = tb->pc;
    dc->tb = tb;
    gen_opc_ptr = gen_opc_buf;
    gen_opc_end = gen_opc_buf + OPC_MAX_SIZE;
    gen_opparam_ptr = gen_opparam_buf;
    dc->is_jmp = DISAS_NEXT;
    dc->pc = pc_start;
    dc->singlestep_enabled = env->singlestep_enabled;
    dc->condjmp = 0;
    dc->thumb = env->thumb;
    dc->is_mem = 0;
#if !defined(CONFIG_USER_ONLY)
    dc->user = (env->uncached_cpsr & 0x1f) == ARM_CPU_MODE_USR;
#endif
    next_page_start = (pc_start & TARGET_PAGE_MASK) + TARGET_PAGE_SIZE;
    nb_gen_labels = 0;
    lj = -1;
    do {
        if (env->nb_breakpoints > 0) {
            for(j = 0; j < env->nb_breakpoints; j++) {
                if (env->breakpoints[j] == dc->pc) {
                    gen_op_movl_T0_im((long)dc->pc);
                    gen_op_movl_reg_TN[0][15]();
                    gen_op_debug();
                    dc->is_jmp = DISAS_JUMP;
                }
            }
        }
        if (search_pc) {
            j = gen_opc_ptr - gen_opc_buf;
            if (lj < j) {
                lj++;
                while (lj < j)
                    gen_opc_instr_start[lj++] = 0;
            }
            gen_opc_pc[lj] = dc->pc;
            gen_opc_instr_start[lj] = 1;
        }
        if (env->thumb)
          disas_thumb_insn(dc);
        else
          disas_arm_insn(env, dc);
        if (dc->condjmp && !dc->is_jmp) {
            gen_set_label(dc->condlabel);
            dc->condjmp = 0;
        }
        /* Translation stops when a conditional branch is enoutered.
         * Otherwise the subsequent code could get translated several times.
         * Also stop translation when a page boundary is reached.  This
         * ensures prefech aborts occur at the right place.  */
    } while (!dc->is_jmp && gen_opc_ptr < gen_opc_end &&
             !env->singlestep_enabled &&
             dc->pc < next_page_start);
    /* At this stage dc->condjmp will only be set when the skipped
     * instruction was a conditional branch, and the PC has already been
     * written.  */
    if (__builtin_expect(env->singlestep_enabled, 0)) {
        /* Make sure the pc is updated, and raise a debug exception.  */
        if (dc->condjmp) {
            gen_op_debug();
            gen_set_label(dc->condlabel);
        }
        if (dc->condjmp || !dc->is_jmp) {
            gen_op_movl_T0_im((long)dc->pc);
            gen_op_movl_reg_TN[0][15]();
            dc->condjmp = 0;
        }
        gen_op_debug();
    } else {
        switch(dc->is_jmp) {
        case DISAS_NEXT:
            gen_goto_tb(dc, 1, dc->pc);
        default:
        case DISAS_JUMP:
        case DISAS_UPDATE:
            /* indicate that the hash table must be used to find the next TB */
            gen_op_movl_T0_0();
            gen_op_exit_tb();
        case DISAS_TB_JUMP:
            /* nothing more to generate */
        }
        if (dc->condjmp) {
            gen_set_label(dc->condlabel);
            gen_goto_tb(dc, 1, dc->pc);
            dc->condjmp = 0;
        }
    }
    *gen_opc_ptr = INDEX_op_end;
#ifdef DEBUG_DISAS
    if (loglevel & CPU_LOG_TB_IN_ASM) {
        fprintf(logfile, "----------------\n");
        fprintf(logfile, "IN: %s\n", lookup_symbol(pc_start));
        target_disas(logfile, pc_start, dc->pc - pc_start, env->thumb);
        fprintf(logfile, "\n");
        if (loglevel & (CPU_LOG_TB_OP)) {
            fprintf(logfile, "OP:\n");
            dump_ops(gen_opc_buf, gen_opparam_buf);
            fprintf(logfile, "\n");
        }
    }
#endif
    if (search_pc) {
        j = gen_opc_ptr - gen_opc_buf;
        lj++;
        while (lj <= j)
            gen_opc_instr_start[lj++] = 0;
        tb->size = 0;
    } else {
        tb->size = dc->pc - pc_start;
    }
    return 0;
}