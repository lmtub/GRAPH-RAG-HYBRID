restore_sigcontext(CPUMIPSState *regs, struct target_sigcontext *sc)

{

    int err = 0;

    int i;



    __get_user(regs->CP0_EPC, &sc->sc_pc);



    __get_user(regs->active_tc.HI[0], &sc->sc_mdhi);

    __get_user(regs->active_tc.LO[0], &sc->sc_mdlo);



    for (i = 1; i < 32; ++i) {

        __get_user(regs->active_tc.gpr[i], &sc->sc_regs[i]);

    }



    __get_user(regs->active_tc.HI[1], &sc->sc_hi1);

    __get_user(regs->active_tc.HI[2], &sc->sc_hi2);

    __get_user(regs->active_tc.HI[3], &sc->sc_hi3);

    __get_user(regs->active_tc.LO[1], &sc->sc_lo1);

    __get_user(regs->active_tc.LO[2], &sc->sc_lo2);

    __get_user(regs->active_tc.LO[3], &sc->sc_lo3);

    {

        uint32_t dsp;

        __get_user(dsp, &sc->sc_dsp);

        cpu_wrdsp(dsp, 0x3ff, regs);

    }



    for (i = 0; i < 32; ++i) {

        __get_user(regs->active_fpu.fpr[i].d, &sc->sc_fpregs[i]);

    }



    return err;

}
