void s390_program_interrupt(CPUS390XState *env, uint32_t code, int ilen,

                            uintptr_t ra)

{

#ifdef CONFIG_TCG

    S390CPU *cpu = s390_env_get_cpu(env);



    if (tcg_enabled()) {

        cpu_restore_state(CPU(cpu), ra);

    }

#endif

    program_interrupt(env, code, ilen);

}
