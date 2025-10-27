setup_sigcontext(CPUMIPSState *regs, struct target_sigcontext *sc)

{

    int err = 0;

    int i;



    __put_user(exception_resume_pc(regs), &sc->sc_pc);

    regs->hflags &= ~MIPS_HFLAG_BMASK;



    __put_user(0, &sc->sc_regs[0]);

    for (i = 1; i < 32; ++i) {

        __put_user(regs->active_tc.gpr[i], &sc->sc_regs[i]);

    }



    __put_user(regs->active_tc.HI[0], &sc->sc_mdhi);

    __put_user(regs->active_tc.LO[0], &sc->sc_mdlo);



    /* Rather than checking for dsp existence, always copy.  The storage

       would just be garbage otherwise.  */

    __put_user(regs->active_tc.HI[1], &sc->sc_hi1);

    __put_user(regs->active_tc.HI[2], &sc->sc_hi2);

    __put_user(regs->active_tc.HI[3], &sc->sc_hi3);

    __put_user(regs->active_tc.LO[1], &sc->sc_lo1);

    __put_user(regs->active_tc.LO[2], &sc->sc_lo2);

    __put_user(regs->active_tc.LO[3], &sc->sc_lo3);

    {

        uint32_t dsp = cpu_rddsp(0x3ff, regs);

        __put_user(dsp, &sc->sc_dsp);

    }



    __put_user(1, &sc->sc_used_math);



    for (i = 0; i < 32; ++i) {

        __put_user(regs->active_fpu.fpr[i].d, &sc->sc_fpregs[i]);

    }



    return err;

}
