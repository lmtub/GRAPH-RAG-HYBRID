static void load_tc(QEMUFile *f, TCState *tc)

{

    int i;



    /* Save active TC */

    for(i = 0; i < 32; i++)

        qemu_get_betls(f, &tc->gpr[i]);

    qemu_get_betls(f, &tc->PC);

    for(i = 0; i < MIPS_DSP_ACC; i++)

        qemu_get_betls(f, &tc->HI[i]);

    for(i = 0; i < MIPS_DSP_ACC; i++)

        qemu_get_betls(f, &tc->LO[i]);

    for(i = 0; i < MIPS_DSP_ACC; i++)

        qemu_get_betls(f, &tc->ACX[i]);

    qemu_get_betls(f, &tc->DSPControl);

    qemu_get_sbe32s(f, &tc->CP0_TCStatus);

    qemu_get_sbe32s(f, &tc->CP0_TCBind);

    qemu_get_betls(f, &tc->CP0_TCHalt);

    qemu_get_betls(f, &tc->CP0_TCContext);

    qemu_get_betls(f, &tc->CP0_TCSchedule);

    qemu_get_betls(f, &tc->CP0_TCScheFBack);

    qemu_get_sbe32s(f, &tc->CP0_Debug_tcstatus);

}
