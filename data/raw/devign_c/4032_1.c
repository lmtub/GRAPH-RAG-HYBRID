static void cpu_handle_guest_debug(CPUState *env)

{

    gdb_set_stop_cpu(env);

    qemu_system_debug_request();

#ifdef CONFIG_IOTHREAD

    env->stopped = 1;

#endif

}
