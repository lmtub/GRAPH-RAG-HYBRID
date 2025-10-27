uint32_t HELPER(testblock)(CPUS390XState *env, uint64_t real_addr)

{

    uintptr_t ra = GETPC();

    CPUState *cs = CPU(s390_env_get_cpu(env));

    int i;



    real_addr = wrap_address(env, real_addr) & TARGET_PAGE_MASK;



    /* Check low-address protection */

    if ((env->cregs[0] & CR0_LOWPROT) && real_addr < 0x2000) {

        cpu_restore_state(cs, ra);

        program_interrupt(env, PGM_PROTECTION, 4);

        return 1;

    }



    for (i = 0; i < TARGET_PAGE_SIZE; i += 8) {

        cpu_stq_real_ra(env, real_addr + i, 0, ra);

    }



    return 0;

}
