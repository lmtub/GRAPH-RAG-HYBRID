static int restore_sigcontext(CPUAlphaState *env,

                              struct target_sigcontext *sc)

{

    uint64_t fpcr;

    int i, err = 0;



    __get_user(env->pc, &sc->sc_pc);



    for (i = 0; i < 31; ++i) {

        __get_user(env->ir[i], &sc->sc_regs[i]);

    }

    for (i = 0; i < 31; ++i) {

        __get_user(env->fir[i], &sc->sc_fpregs[i]);

    }



    __get_user(fpcr, &sc->sc_fpcr);

    cpu_alpha_store_fpcr(env, fpcr);



    return err;

}
