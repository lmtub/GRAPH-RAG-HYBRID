void helper_ldxfsr(CPUSPARCState *env, uint64_t new_fsr)

{

    env->fsr = (new_fsr & FSR_LDXFSR_MASK) | (env->fsr & FSR_LDXFSR_OLDMASK);

    set_fsr(env);

}
