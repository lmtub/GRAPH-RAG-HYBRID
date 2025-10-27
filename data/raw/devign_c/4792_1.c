uint64_t HELPER(get_cp_reg64)(CPUARMState *env, void *rip)

{

    const ARMCPRegInfo *ri = rip;



    return ri->readfn(env, ri);

}
