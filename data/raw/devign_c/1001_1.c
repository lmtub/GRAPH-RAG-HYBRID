static void cpu_common_initfn(Object *obj)

{

    CPUState *cpu = CPU(obj);

    CPUClass *cc = CPU_GET_CLASS(obj);



    cpu->cpu_index = UNASSIGNED_CPU_INDEX;

    cpu->gdb_num_regs = cpu->gdb_num_g_regs = cc->gdb_num_core_regs;

    /* *-user doesn't have configurable SMP topology */

    /* the default value is changed by qemu_init_vcpu() for softmmu */

    cpu->nr_cores = 1;

    cpu->nr_threads = 1;



    qemu_mutex_init(&cpu->work_mutex);

    QTAILQ_INIT(&cpu->breakpoints);

    QTAILQ_INIT(&cpu->watchpoints);



    cpu->trace_dstate = bitmap_new(trace_get_vcpu_event_count());



    cpu_exec_initfn(cpu);

}
