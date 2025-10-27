void gen_intermediate_code(CPUState *cs, TranslationBlock *tb)

{

    CPUM68KState *env = cs->env_ptr;

    DisasContext dc1, *dc = &dc1;

    target_ulong pc_start;

    int pc_offset;

    int num_insns;

    int max_insns;



    /* generate intermediate code */

    pc_start = tb->pc;



    dc->tb = tb;



    dc->env = env;

    dc->is_jmp = DISAS_NEXT;

    dc->pc = pc_start;

    dc->cc_op = CC_OP_DYNAMIC;

    dc->cc_op_synced = 1;

    dc->singlestep_enabled = cs->singlestep_enabled;

    dc->user = (env->sr & SR_S) == 0;

    dc->done_mac = 0;

    dc->writeback_mask = 0;

    num_insns = 0;

    max_insns = tb->cflags & CF_COUNT_MASK;

    if (max_insns == 0) {

        max_insns = CF_COUNT_MASK;

    }

    if (max_insns > TCG_MAX_INSNS) {

        max_insns = TCG_MAX_INSNS;

    }



    gen_tb_start(tb);

    do {

        pc_offset = dc->pc - pc_start;

        gen_throws_exception = NULL;

        tcg_gen_insn_start(dc->pc, dc->cc_op);

        num_insns++;



        if (unlikely(cpu_breakpoint_test(cs, dc->pc, BP_ANY))) {

            gen_exception(dc, dc->pc, EXCP_DEBUG);

            dc->is_jmp = DISAS_JUMP;

            /* The address covered by the breakpoint must be included in

               [tb->pc, tb->pc + tb->size) in order to for it to be

               properly cleared -- thus we increment the PC here so that

               the logic setting tb->size below does the right thing.  */

            dc->pc += 2;

            break;

        }



        if (num_insns == max_insns && (tb->cflags & CF_LAST_IO)) {

            gen_io_start();

        }



        dc->insn_pc = dc->pc;

	disas_m68k_insn(env, dc);

    } while (!dc->is_jmp && !tcg_op_buf_full() &&

             !cs->singlestep_enabled &&

             !singlestep &&

             (pc_offset) < (TARGET_PAGE_SIZE - 32) &&

             num_insns < max_insns);



    if (tb->cflags & CF_LAST_IO)

        gen_io_end();

    if (unlikely(cs->singlestep_enabled)) {

        /* Make sure the pc is updated, and raise a debug exception.  */

        if (!dc->is_jmp) {

            update_cc_op(dc);

            tcg_gen_movi_i32(QREG_PC, dc->pc);

        }

        gen_helper_raise_exception(cpu_env, tcg_const_i32(EXCP_DEBUG));

    } else {

        switch(dc->is_jmp) {

        case DISAS_NEXT:

            update_cc_op(dc);

            gen_jmp_tb(dc, 0, dc->pc);

            break;

        default:

        case DISAS_JUMP:

        case DISAS_UPDATE:

            update_cc_op(dc);

            /* indicate that the hash table must be used to find the next TB */

            tcg_gen_exit_tb(0);

            break;

        case DISAS_TB_JUMP:

            /* nothing more to generate */

            break;

        }

    }

    gen_tb_end(tb, num_insns);



#ifdef DEBUG_DISAS

    if (qemu_loglevel_mask(CPU_LOG_TB_IN_ASM)

        && qemu_log_in_addr_range(pc_start)) {

        qemu_log_lock();

        qemu_log("----------------\n");

        qemu_log("IN: %s\n", lookup_symbol(pc_start));

        log_target_disas(cs, pc_start, dc->pc - pc_start, 0);

        qemu_log("\n");

        qemu_log_unlock();

    }

#endif

    tb->size = dc->pc - pc_start;

    tb->icount = num_insns;

}
