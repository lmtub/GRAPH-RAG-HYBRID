void helper_rfmci(CPUPPCState *env)

{

    do_rfi(env, env->spr[SPR_BOOKE_MCSRR0], SPR_BOOKE_MCSRR1,

           ~((target_ulong)0x3FFF0000), 0);

}
