void cpu_exec_init(CPUState *cpu, Error **errp)

{

    CPUClass *cc = CPU_GET_CLASS(cpu);

    int cpu_index;

    Error *local_err = NULL;



#ifndef CONFIG_USER_ONLY

    cpu->as = &address_space_memory;

    cpu->thread_id = qemu_get_thread_id();

#endif



#if defined(CONFIG_USER_ONLY)

    cpu_list_lock();

#endif

    cpu_index = cpu->cpu_index = cpu_get_free_index(&local_err);

    if (local_err) {

        error_propagate(errp, local_err);

#if defined(CONFIG_USER_ONLY)

        cpu_list_unlock();

#endif

        return;

    }

    QTAILQ_INSERT_TAIL(&cpus, cpu, node);

#if defined(CONFIG_USER_ONLY)

    cpu_list_unlock();

#endif

    if (qdev_get_vmsd(DEVICE(cpu)) == NULL) {

        vmstate_register(NULL, cpu_index, &vmstate_cpu_common, cpu);

    }

#if defined(CPU_SAVE_VERSION) && !defined(CONFIG_USER_ONLY)

    register_savevm(NULL, "cpu", cpu_index, CPU_SAVE_VERSION,

                    cpu_save, cpu_load, cpu->env_ptr);

    assert(cc->vmsd == NULL);

    assert(qdev_get_vmsd(DEVICE(cpu)) == NULL);

#endif

    if (cc->vmsd != NULL) {

        vmstate_register(NULL, cpu_index, cc->vmsd, cpu);

    }

}
