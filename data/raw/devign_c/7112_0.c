void s390_init_cpus(const char *cpu_model)

{

    int i;



    if (cpu_model == NULL) {

        cpu_model = "host";

    }



    ipi_states = g_malloc(sizeof(S390CPU *) * smp_cpus);



    for (i = 0; i < smp_cpus; i++) {

        S390CPU *cpu;

        CPUState *cs;



        cpu = cpu_s390x_init(cpu_model);

        cs = CPU(cpu);



        ipi_states[i] = cpu;

        cs->halted = 1;

        cs->exception_index = EXCP_HLT;

    }

}
