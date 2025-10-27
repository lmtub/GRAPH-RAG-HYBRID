static inline int nvic_exec_prio(NVICState *s)

{

    CPUARMState *env = &s->cpu->env;

    int running;



    if (env->daif & PSTATE_F) { /* FAULTMASK */

        running = -1;

    } else if (env->daif & PSTATE_I) { /* PRIMASK */

        running = 0;

    } else if (env->v7m.basepri > 0) {

        running = env->v7m.basepri & nvic_gprio_mask(s);

    } else {

        running = NVIC_NOEXC_PRIO; /* lower than any possible priority */

    }

    /* consider priority of active handler */

    return MIN(running, s->exception_prio);

}
