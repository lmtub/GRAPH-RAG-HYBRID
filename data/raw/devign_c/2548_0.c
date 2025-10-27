static int handle_instruction(S390CPU *cpu, struct kvm_run *run)

{

    unsigned int ipa0 = (run->s390_sieic.ipa & 0xff00);

    uint8_t ipa1 = run->s390_sieic.ipa & 0x00ff;

    int r = -1;



    DPRINTF("handle_instruction 0x%x 0x%x\n",

            run->s390_sieic.ipa, run->s390_sieic.ipb);

    switch (ipa0) {

    case IPA0_B2:

        r = handle_b2(cpu, run, ipa1);

        break;

    case IPA0_B9:

        r = handle_b9(cpu, run, ipa1);

        break;

    case IPA0_EB:

        r = handle_eb(cpu, run, run->s390_sieic.ipb & 0xff);

        break;

    case IPA0_E3:

        r = handle_e3(cpu, run, run->s390_sieic.ipb & 0xff);

        break;

    case IPA0_DIAG:

        r = handle_diag(cpu, run, run->s390_sieic.ipb);

        break;

    case IPA0_SIGP:

        r = handle_sigp(cpu, ipa1, run->s390_sieic.ipb);

        break;

    }



    if (r < 0) {

        r = 0;

        kvm_s390_program_interrupt(cpu, PGM_OPERATION);

    }



    return r;

}
