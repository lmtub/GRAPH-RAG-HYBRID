void cpu_exec_init(CPUArchState *env)

{

    CPUState *cpu = ENV_GET_CPU(env);

    CPUClass *cc = CPU_GET_CLASS(cpu);

    CPUState *some_cpu;

    int cpu_index;



#if defined(CONFIG_USER_ONLY)

    cpu_list_lock();

#endif

    cpu_index = 0;

    CPU_FOREACH(some_cpu) {

        cpu_index++;

    }

    cpu->cpu_index = cpu_index;

    cpu->numa_node = 0;

    QTAILQ_INIT(&cpu->breakpoints);

    QTAILQ_INIT(&cpu->watchpoints);

#ifndef CONFIG_USER_ONLY

    cpu->as = &address_space_memory;

    cpu->thread_id = qemu_get_thread_id();


#endif

    QTAILQ_INSERT_TAIL(&cpus, cpu, node);

#if defined(CONFIG_USER_ONLY)

    cpu_list_unlock();

#endif

    if (qdev_get_vmsd(DEVICE(cpu)) == NULL) {

        vmstate_register(NULL, cpu_index, &vmstate_cpu_common, cpu);

    }

#if defined(CPU_SAVE_VERSION) && !defined(CONFIG_USER_ONLY)

    register_savevm(NULL, "cpu", cpu_index, CPU_SAVE_VERSION,

                    cpu_save, cpu_load, env);

    assert(cc->vmsd == NULL);

    assert(qdev_get_vmsd(DEVICE(cpu)) == NULL);

#endif

    if (cc->vmsd != NULL) {

        vmstate_register(NULL, cpu_index, cc->vmsd, cpu);

    }

}