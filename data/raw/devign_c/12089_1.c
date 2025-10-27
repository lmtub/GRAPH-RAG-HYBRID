restore_sigcontext(CPUM68KState *env, struct target_sigcontext *sc, int *pd0)

{

    int err = 0;

    int temp;



    __get_user(env->aregs[7], &sc->sc_usp);

    __get_user(env->dregs[1], &sc->sc_d1);

    __get_user(env->aregs[0], &sc->sc_a0);

    __get_user(env->aregs[1], &sc->sc_a1);

    __get_user(env->pc, &sc->sc_pc);

    __get_user(temp, &sc->sc_sr);

    env->sr = (env->sr & 0xff00) | (temp & 0xff);



    *pd0 = tswapl(sc->sc_d0);



    return err;

}
