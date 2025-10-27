static int restore_sigcontext(CPUSH4State *regs, struct target_sigcontext *sc,

                              target_ulong *r0_p)

{

    unsigned int err = 0;

    int i;



#define COPY(x)         __get_user(regs->x, &sc->sc_##x)

    COPY(gregs[1]);

    COPY(gregs[2]); COPY(gregs[3]);

    COPY(gregs[4]); COPY(gregs[5]);

    COPY(gregs[6]); COPY(gregs[7]);

    COPY(gregs[8]); COPY(gregs[9]);

    COPY(gregs[10]); COPY(gregs[11]);

    COPY(gregs[12]); COPY(gregs[13]);

    COPY(gregs[14]); COPY(gregs[15]);

    COPY(gbr); COPY(mach);

    COPY(macl); COPY(pr);

    COPY(sr); COPY(pc);

#undef COPY



    for (i=0; i<16; i++) {

        __get_user(regs->fregs[i], &sc->sc_fpregs[i]);

    }

    __get_user(regs->fpscr, &sc->sc_fpscr);

    __get_user(regs->fpul, &sc->sc_fpul);



    regs->tra = -1;         /* disable syscall checks */

    __get_user(*r0_p, &sc->sc_gregs[0]);

    return err;

}
