int use_gdb_syscalls(void)

{

    if (gdb_syscall_mode == GDB_SYS_UNKNOWN) {

        gdb_syscall_mode = (gdb_syscall_state ? GDB_SYS_ENABLED

                                              : GDB_SYS_DISABLED);

    }

    return gdb_syscall_mode == GDB_SYS_ENABLED;

}
