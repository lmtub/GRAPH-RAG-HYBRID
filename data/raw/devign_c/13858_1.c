void QEMU_NORETURN do_unassigned_access(target_phys_addr_t addr, int is_write,

                                        int is_exec, int unused, int size)

{

    env->trap_arg0 = addr;

    env->trap_arg1 = is_write;

    dynamic_excp(EXCP_MCHK, 0);

}
