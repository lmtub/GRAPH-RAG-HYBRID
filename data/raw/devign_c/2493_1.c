unsigned int s390_cpu_set_state(uint8_t cpu_state, S390CPU *cpu)

 {

    trace_cpu_set_state(CPU(cpu)->cpu_index, cpu_state);



    switch (cpu_state) {

    case CPU_STATE_STOPPED:

    case CPU_STATE_CHECK_STOP:

        /* halt the cpu for common infrastructure */

        s390_cpu_halt(cpu);

        break;

    case CPU_STATE_OPERATING:

    case CPU_STATE_LOAD:

        /* unhalt the cpu for common infrastructure */

        s390_cpu_unhalt(cpu);

        break;

    default:

        error_report("Requested CPU state is not a valid S390 CPU state: %u",

                     cpu_state);

        exit(1);





    cpu->env.cpu_state = cpu_state;



    return s390_count_running_cpus();
