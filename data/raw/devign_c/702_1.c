void helper_rfdi(CPUPPCState *env)

{

    do_rfi(env, env->spr[SPR_BOOKE_DSRR0], SPR_BOOKE_DSRR1,

           ~((target_ulong)0x3FFF0000), 0);

}
