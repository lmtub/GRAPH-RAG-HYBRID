static int gdb_breakpoint_insert(CPUState *env, target_ulong addr,

                                 target_ulong len, int type)

{

    switch (type) {

    case GDB_BREAKPOINT_SW:

    case GDB_BREAKPOINT_HW:

        return cpu_breakpoint_insert(env, addr, BP_GDB, NULL);

#ifndef CONFIG_USER_ONLY

    case GDB_WATCHPOINT_WRITE:

    case GDB_WATCHPOINT_READ:

    case GDB_WATCHPOINT_ACCESS:

        return cpu_watchpoint_insert(env, addr, len, xlat_gdb_type[type],

                                     NULL);

#endif

    default:

        return -ENOSYS;

    }

}
