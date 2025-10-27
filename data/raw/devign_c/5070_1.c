void cpsr_write(CPUARMState *env, uint32_t val, uint32_t mask)

{

    if (mask & CPSR_NZCV) {

        env->ZF = (~val) & CPSR_Z;

        env->NF = val;

        env->CF = (val >> 29) & 1;

        env->VF = (val << 3) & 0x80000000;

    }

    if (mask & CPSR_Q)

        env->QF = ((val & CPSR_Q) != 0);

    if (mask & CPSR_T)

        env->thumb = ((val & CPSR_T) != 0);

    if (mask & CPSR_IT_0_1) {

        env->condexec_bits &= ~3;

        env->condexec_bits |= (val >> 25) & 3;

    }

    if (mask & CPSR_IT_2_7) {

        env->condexec_bits &= 3;

        env->condexec_bits |= (val >> 8) & 0xfc;

    }

    if (mask & CPSR_GE) {

        env->GE = (val >> 16) & 0xf;

    }



    if ((env->uncached_cpsr ^ val) & mask & CPSR_M) {

        switch_mode(env, val & CPSR_M);

    }

    mask &= ~CACHED_CPSR_BITS;

    env->uncached_cpsr = (env->uncached_cpsr & ~mask) | (val & mask);

}
