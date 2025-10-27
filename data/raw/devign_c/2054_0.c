void cpu_x86_load_seg(CPUX86State *s, int seg_reg, int selector)

{

    CPUX86State *saved_env;



    saved_env = env;

    env = s;

    if (env->eflags & VM_MASK) {

        SegmentCache *sc;

        selector &= 0xffff;

        sc = &env->seg_cache[seg_reg];

        /* NOTE: in VM86 mode, limit and seg_32bit are never reloaded,

           so we must load them here */

        sc->base = (void *)(selector << 4);

        sc->limit = 0xffff;

        sc->seg_32bit = 0;

        env->segs[seg_reg] = selector;

    } else {

        load_seg(seg_reg, selector, 0);

    }

    env = saved_env;

}
