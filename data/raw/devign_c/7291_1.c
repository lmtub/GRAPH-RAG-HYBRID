void do_unassigned_access(target_phys_addr_t addr, int is_write, int is_exec,

                          int is_asi, int size)

{

    CPUState *saved_env;

    int fault_type;



    /* XXX: hack to restore env in all cases, even if not called from

       generated code */

    saved_env = env;

    env = cpu_single_env;

#ifdef DEBUG_UNASSIGNED

    if (is_asi)

        printf("Unassigned mem %s access of %d byte%s to " TARGET_FMT_plx

               " asi 0x%02x from " TARGET_FMT_lx "\n",

               is_exec ? "exec" : is_write ? "write" : "read", size,

               size == 1 ? "" : "s", addr, is_asi, env->pc);

    else

        printf("Unassigned mem %s access of %d byte%s to " TARGET_FMT_plx

               " from " TARGET_FMT_lx "\n",

               is_exec ? "exec" : is_write ? "write" : "read", size,

               size == 1 ? "" : "s", addr, env->pc);

#endif

    /* Don't overwrite translation and access faults */

    fault_type = (env->mmuregs[3] & 0x1c) >> 2;

    if ((fault_type > 4) || (fault_type == 0)) {

        env->mmuregs[3] = 0; /* Fault status register */

        if (is_asi)

            env->mmuregs[3] |= 1 << 16;

        if (env->psrs)

            env->mmuregs[3] |= 1 << 5;

        if (is_exec)

            env->mmuregs[3] |= 1 << 6;

        if (is_write)

            env->mmuregs[3] |= 1 << 7;

        env->mmuregs[3] |= (5 << 2) | 2;

        /* SuperSPARC will never place instruction fault addresses in the FAR */

        if (!is_exec) {

            env->mmuregs[4] = addr; /* Fault address register */

        }

    }

    /* overflow (same type fault was not read before another fault) */

    if (fault_type == ((env->mmuregs[3] & 0x1c)) >> 2) {

        env->mmuregs[3] |= 1;

    }



    if ((env->mmuregs[0] & MMU_E) && !(env->mmuregs[0] & MMU_NF)) {

        if (is_exec)

            raise_exception(TT_CODE_ACCESS);

        else

            raise_exception(TT_DATA_ACCESS);

    }



    /* flush neverland mappings created during no-fault mode,

       so the sequential MMU faults report proper fault types */

    if (env->mmuregs[0] & MMU_NF) {

        tlb_flush(env, 1);

    }



    env = saved_env;

}
