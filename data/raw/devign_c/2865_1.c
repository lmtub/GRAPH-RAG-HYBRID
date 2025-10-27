static int modified_clear_reset(S390CPU *cpu)

{

    S390CPUClass *scc = S390_CPU_GET_CLASS(cpu);



    pause_all_vcpus();

    cpu_synchronize_all_states();

    cpu_full_reset_all();


    io_subsystem_reset();

    scc->load_normal(CPU(cpu));

    cpu_synchronize_all_post_reset();

    resume_all_vcpus();

    return 0;

}