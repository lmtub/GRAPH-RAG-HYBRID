void helper_ldfsr(CPUSPARCState *env, uint32_t new_fsr)

{

    env->fsr = (new_fsr & FSR_LDFSR_MASK) | (env->fsr & FSR_LDFSR_OLDMASK);

    set_fsr(env);

}
