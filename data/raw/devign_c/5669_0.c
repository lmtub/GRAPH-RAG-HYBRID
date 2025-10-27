void cpu_exec_init_all(void)

{

#if !defined(CONFIG_USER_ONLY)

    qemu_mutex_init(&ram_list.mutex);

    memory_map_init();

    io_mem_init();

#endif

}
