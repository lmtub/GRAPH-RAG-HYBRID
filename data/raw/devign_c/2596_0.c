static inline void check_alignment(CPUS390XState *env, uint64_t v,

                                   int wordsize, uintptr_t ra)

{

    if (v % wordsize) {

        CPUState *cs = CPU(s390_env_get_cpu(env));

        cpu_restore_state(cs, ra);

        program_interrupt(env, PGM_SPECIFICATION, 6);

    }

}
