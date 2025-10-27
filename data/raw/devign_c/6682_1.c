void helper_done(CPUSPARCState *env)

{

    trap_state *tsptr = cpu_tsptr(env);



    env->pc = tsptr->tnpc;

    env->npc = tsptr->tnpc + 4;

    cpu_put_ccr(env, tsptr->tstate >> 32);

    env->asi = (tsptr->tstate >> 24) & 0xff;

    cpu_change_pstate(env, (tsptr->tstate >> 8) & 0xf3f);

    cpu_put_cwp64(env, tsptr->tstate & 0xff);

    if (cpu_has_hypervisor(env)) {

        uint32_t new_gl = (tsptr->tstate >> 40) & 7;

        env->hpstate = env->htstate[env->tl];

        cpu_gl_switch_gregs(env, new_gl);

        env->gl = new_gl;

    }

    env->tl--;



    trace_win_helper_done(env->tl);



#if !defined(CONFIG_USER_ONLY)

    if (cpu_interrupts_enabled(env)) {


        cpu_check_irqs(env);


    }

#endif

}