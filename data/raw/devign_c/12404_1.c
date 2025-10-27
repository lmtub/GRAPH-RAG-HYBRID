static int mpidr_read(CPUARMState *env, const ARMCPRegInfo *ri,

                      uint64_t *value)

{

    CPUState *cs = CPU(arm_env_get_cpu(env));

    uint32_t mpidr = cs->cpu_index;

    /* We don't support setting cluster ID ([8..11])

     * so these bits always RAZ.

     */

    if (arm_feature(env, ARM_FEATURE_V7MP)) {

        mpidr |= (1 << 31);

        /* Cores which are uniprocessor (non-coherent)

         * but still implement the MP extensions set

         * bit 30. (For instance, A9UP.) However we do

         * not currently model any of those cores.

         */

    }

    *value = mpidr;

    return 0;

}
