int kvmppc_put_books_sregs(PowerPCCPU *cpu)

{

    CPUPPCState *env = &cpu->env;

    struct kvm_sregs sregs;

    int i;



    sregs.pvr = env->spr[SPR_PVR];



    sregs.u.s.sdr1 = env->spr[SPR_SDR1];



    /* Sync SLB */

#ifdef TARGET_PPC64

    for (i = 0; i < ARRAY_SIZE(env->slb); i++) {

        sregs.u.s.ppc64.slb[i].slbe = env->slb[i].esid;

        if (env->slb[i].esid & SLB_ESID_V) {

            sregs.u.s.ppc64.slb[i].slbe |= i;

        }

        sregs.u.s.ppc64.slb[i].slbv = env->slb[i].vsid;

    }

#endif



    /* Sync SRs */

    for (i = 0; i < 16; i++) {

        sregs.u.s.ppc32.sr[i] = env->sr[i];

    }



    /* Sync BATs */

    for (i = 0; i < 8; i++) {

        /* Beware. We have to swap upper and lower bits here */

        sregs.u.s.ppc32.dbat[i] = ((uint64_t)env->DBAT[0][i] << 32)

            | env->DBAT[1][i];

        sregs.u.s.ppc32.ibat[i] = ((uint64_t)env->IBAT[0][i] << 32)

            | env->IBAT[1][i];

    }



    return kvm_vcpu_ioctl(CPU(cpu), KVM_SET_SREGS, &sregs);

}
