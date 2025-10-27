static inline tcg_target_ulong cpu_tb_exec(CPUState *cpu, TranslationBlock *itb)

{

    CPUArchState *env = cpu->env_ptr;

    uintptr_t ret;

    TranslationBlock *last_tb;

    int tb_exit;

    uint8_t *tb_ptr = itb->tc_ptr;



    qemu_log_mask_and_addr(CPU_LOG_EXEC, itb->pc,

                           "Trace %p [" TARGET_FMT_lx "] %s\n",

                           itb->tc_ptr, itb->pc, lookup_symbol(itb->pc));



#if defined(DEBUG_DISAS)

    if (qemu_loglevel_mask(CPU_LOG_TB_CPU)

        && qemu_log_in_addr_range(itb->pc)) {

#if defined(TARGET_I386)

        log_cpu_state(cpu, CPU_DUMP_CCOP);

#elif defined(TARGET_M68K)

        /* ??? Should not modify env state for dumping.  */

        cpu_m68k_flush_flags(env, env->cc_op);

        env->cc_op = CC_OP_FLAGS;

        env->sr = (env->sr & 0xffe0) | env->cc_dest | (env->cc_x << 4);

        log_cpu_state(cpu, 0);

#else

        log_cpu_state(cpu, 0);

#endif

    }

#endif /* DEBUG_DISAS */



    cpu->can_do_io = !use_icount;

    ret = tcg_qemu_tb_exec(env, tb_ptr);

    cpu->can_do_io = 1;

    last_tb = (TranslationBlock *)(ret & ~TB_EXIT_MASK);

    tb_exit = ret & TB_EXIT_MASK;

    trace_exec_tb_exit(last_tb, tb_exit);



    if (tb_exit > TB_EXIT_IDX1) {

        /* We didn't start executing this TB (eg because the instruction

         * counter hit zero); we must restore the guest PC to the address

         * of the start of the TB.

         */

        CPUClass *cc = CPU_GET_CLASS(cpu);

        qemu_log_mask_and_addr(CPU_LOG_EXEC, last_tb->pc,

                               "Stopped execution of TB chain before %p ["

                               TARGET_FMT_lx "] %s\n",

                               last_tb->tc_ptr, last_tb->pc,

                               lookup_symbol(last_tb->pc));

        if (cc->synchronize_from_tb) {

            cc->synchronize_from_tb(cpu, last_tb);

        } else {

            assert(cc->set_pc);

            cc->set_pc(cpu, last_tb->pc);

        }

    }

    if (tb_exit == TB_EXIT_REQUESTED) {

        /* We were asked to stop executing TBs (probably a pending

         * interrupt. We've now stopped, so clear the flag.

         */

        cpu->tcg_exit_req = 0;

    }

    return ret;

}
