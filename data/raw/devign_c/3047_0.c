void cpu_x86_set_a20(CPUX86State *env, int a20_state)

{

    a20_state = (a20_state != 0);

    if (a20_state != a20_enabled) {

#if defined(DEBUG_MMU)

        printf("A20 update: a20=%d\n", a20_state);

#endif

        /* if the cpu is currently executing code, we must unlink it and

           all the potentially executing TB */

        cpu_interrupt(env, 0);



        /* when a20 is changed, all the MMU mappings are invalid, so

           we must flush everything */

        tlb_flush(env);

        a20_enabled = a20_state;

        if (a20_enabled)

            a20_mask = 0xffffffff;

        else

            a20_mask = 0xffefffff;

    }

}
