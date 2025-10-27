void helper_retry(void)

{

    env->pc = env->tsptr->tpc;

    env->npc = env->tsptr->tnpc;

    PUT_CCR(env, env->tsptr->tstate >> 32);

    env->asi = (env->tsptr->tstate >> 24) & 0xff;

    change_pstate((env->tsptr->tstate >> 8) & 0xf3f);

    PUT_CWP64(env, env->tsptr->tstate & 0xff);

    env->tl--;

    env->tsptr = &env->ts[env->tl & MAXTL_MASK];

}
