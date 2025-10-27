void armv7m_nvic_acknowledge_irq(void *opaque)

{

    NVICState *s = (NVICState *)opaque;

    CPUARMState *env = &s->cpu->env;

    const int pending = s->vectpending;

    const int running = nvic_exec_prio(s);

    int pendgroupprio;

    VecInfo *vec;



    assert(pending > ARMV7M_EXCP_RESET && pending < s->num_irq);



    vec = &s->vectors[pending];



    assert(vec->enabled);

    assert(vec->pending);



    pendgroupprio = vec->prio;

    if (pendgroupprio > 0) {

        pendgroupprio &= nvic_gprio_mask(s);

    }

    assert(pendgroupprio < running);



    trace_nvic_acknowledge_irq(pending, vec->prio);



    vec->active = 1;

    vec->pending = 0;



    env->v7m.exception = s->vectpending;



    nvic_irq_update(s);

}
