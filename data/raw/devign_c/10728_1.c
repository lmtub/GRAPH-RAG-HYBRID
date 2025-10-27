uint32_t HELPER(get_cp_reg)(CPUARMState *env, void *rip)

{

    const ARMCPRegInfo *ri = rip;



    return ri->readfn(env, ri);

}
