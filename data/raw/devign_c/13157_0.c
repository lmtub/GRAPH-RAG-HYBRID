void cpu_exec_realizefn(CPUState *cpu, Error **errp)

{

    CPUClass *cc = CPU_GET_CLASS(cpu);



    cpu_list_add(cpu);



    if (tcg_enabled() && !cc->tcg_initialized) {

        cc->tcg_initialized = true;

        cc->tcg_initialize();

    }



#ifndef CONFIG_USER_ONLY

    if (qdev_get_vmsd(DEVICE(cpu)) == NULL) {

        vmstate_register(NULL, cpu->cpu_index, &vmstate_cpu_common, cpu);

    }

    if (cc->vmsd != NULL) {

        vmstate_register(NULL, cpu->cpu_index, cc->vmsd, cpu);

    }

#endif

}
