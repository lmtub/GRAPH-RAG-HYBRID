void tlb_fill(unsigned long addr, int is_write, int is_user, void *retaddr)

{

    TranslationBlock *tb;

    CPUState *saved_env;

    unsigned long pc;

    int ret;



    /* XXX: hack to restore env in all cases, even if not called from

       generated code */

    saved_env = env;

    env = cpu_single_env;

    {

        unsigned long tlb_addrr, tlb_addrw;

        int index;

        index = (addr >> TARGET_PAGE_BITS) & (CPU_TLB_SIZE - 1);

        tlb_addrr = env->tlb_read[is_user][index].address;

        tlb_addrw = env->tlb_write[is_user][index].address;

#if 0

        if (loglevel) {

            fprintf(logfile,

                    "%s 1 %p %p idx=%d addr=0x%08lx tbl_addr=0x%08lx 0x%08lx "

               "(0x%08lx 0x%08lx)\n", __func__, env,

               &env->tlb_read[is_user][index], index, addr,

               tlb_addrr, tlb_addrw, addr & TARGET_PAGE_MASK,

               tlb_addrr & (TARGET_PAGE_MASK | TLB_INVALID_MASK));

        }

#endif

    }

    ret = cpu_ppc_handle_mmu_fault(env, addr, is_write, is_user, 1);

    if (ret) {

        if (retaddr) {

            /* now we have a real cpu fault */

            pc = (unsigned long)retaddr;

            tb = tb_find_pc(pc);

            if (tb) {

                /* the PC is inside the translated code. It means that we have

                   a virtual CPU fault */

                cpu_restore_state(tb, env, pc, NULL);

            }

        }

        do_raise_exception_err(env->exception_index, env->error_code);

    }

    {

        unsigned long tlb_addrr, tlb_addrw;

        int index;

        index = (addr >> TARGET_PAGE_BITS) & (CPU_TLB_SIZE - 1);

        tlb_addrr = env->tlb_read[is_user][index].address;

        tlb_addrw = env->tlb_write[is_user][index].address;

#if 0

        printf("%s 2 %p %p idx=%d addr=0x%08lx tbl_addr=0x%08lx 0x%08lx "

               "(0x%08lx 0x%08lx)\n", __func__, env,

               &env->tlb_read[is_user][index], index, addr,

               tlb_addrr, tlb_addrw, addr & TARGET_PAGE_MASK,

               tlb_addrr & (TARGET_PAGE_MASK | TLB_INVALID_MASK));

#endif

    }

    env = saved_env;

}
