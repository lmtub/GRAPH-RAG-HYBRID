static bool tb_invalidate_phys_page(tb_page_addr_t addr, uintptr_t pc)

{

    TranslationBlock *tb;

    PageDesc *p;

    int n;

#ifdef TARGET_HAS_PRECISE_SMC

    TranslationBlock *current_tb = NULL;

    CPUState *cpu = current_cpu;

    CPUArchState *env = NULL;

    int current_tb_modified = 0;

    target_ulong current_pc = 0;

    target_ulong current_cs_base = 0;

    uint32_t current_flags = 0;

#endif



    assert_memory_lock();



    addr &= TARGET_PAGE_MASK;

    p = page_find(addr >> TARGET_PAGE_BITS);

    if (!p) {

        return false;

    }



    tb_lock();

    tb = p->first_tb;

#ifdef TARGET_HAS_PRECISE_SMC

    if (tb && pc != 0) {

        current_tb = tb_find_pc(pc);

    }

    if (cpu != NULL) {

        env = cpu->env_ptr;

    }

#endif

    while (tb != NULL) {

        n = (uintptr_t)tb & 3;

        tb = (TranslationBlock *)((uintptr_t)tb & ~3);

#ifdef TARGET_HAS_PRECISE_SMC

        if (current_tb == tb &&

            (current_tb->cflags & CF_COUNT_MASK) != 1) {

                /* If we are modifying the current TB, we must stop

                   its execution. We could be more precise by checking

                   that the modification is after the current PC, but it

                   would require a specialized function to partially

                   restore the CPU state */



            current_tb_modified = 1;

            cpu_restore_state_from_tb(cpu, current_tb, pc);

            cpu_get_tb_cpu_state(env, &current_pc, &current_cs_base,

                                 &current_flags);

        }

#endif /* TARGET_HAS_PRECISE_SMC */

        tb_phys_invalidate(tb, addr);

        tb = tb->page_next[n];

    }

    p->first_tb = NULL;

#ifdef TARGET_HAS_PRECISE_SMC

    if (current_tb_modified) {

        /* we generate a block containing just the instruction

           modifying the memory. It will ensure that it cannot modify

           itself */

        tb_gen_code(cpu, current_pc, current_cs_base, current_flags,

                    1 | curr_cflags());

        /* tb_lock will be reset after cpu_loop_exit_noexc longjmps

         * back into the cpu_exec loop. */

        return true;

    }

#endif

    tb_unlock();



    return false;

}
