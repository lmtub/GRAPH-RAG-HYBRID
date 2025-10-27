void HELPER(set_cp_reg)(CPUARMState *env, void *rip, uint32_t value)

{

    const ARMCPRegInfo *ri = rip;



    ri->writefn(env, ri, value);

}
