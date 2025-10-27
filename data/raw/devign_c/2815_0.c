int kvm_arch_get_registers(CPUState *env)

{

    struct kvm_regs regs;

    struct kvm_sregs sregs;

    int i, ret;



    ret = kvm_vcpu_ioctl(env, KVM_GET_REGS, &regs);

    if (ret < 0)

        return ret;



    ret = kvm_vcpu_ioctl(env, KVM_GET_SREGS, &sregs);

    if (ret < 0)

        return ret;



    env->ctr = regs.ctr;

    env->lr = regs.lr;

    env->xer = regs.xer;

    env->msr = regs.msr;

    env->nip = regs.pc;



    env->spr[SPR_SRR0] = regs.srr0;

    env->spr[SPR_SRR1] = regs.srr1;



    env->spr[SPR_SPRG0] = regs.sprg0;

    env->spr[SPR_SPRG1] = regs.sprg1;

    env->spr[SPR_SPRG2] = regs.sprg2;

    env->spr[SPR_SPRG3] = regs.sprg3;

    env->spr[SPR_SPRG4] = regs.sprg4;

    env->spr[SPR_SPRG5] = regs.sprg5;

    env->spr[SPR_SPRG6] = regs.sprg6;

    env->spr[SPR_SPRG7] = regs.sprg7;



    for (i = 0;i < 32; i++)

        env->gpr[i] = regs.gpr[i];



#ifdef KVM_CAP_PPC_SEGSTATE

    if (kvm_check_extension(env->kvm_state, KVM_CAP_PPC_SEGSTATE)) {

        env->sdr1 = sregs.u.s.sdr1;



        /* Sync SLB */

#ifdef TARGET_PPC64

        for (i = 0; i < 64; i++) {

            ppc_store_slb(env, sregs.u.s.ppc64.slb[i].slbe,

                               sregs.u.s.ppc64.slb[i].slbv);

        }

#endif



        /* Sync SRs */

        for (i = 0; i < 16; i++) {

            env->sr[i] = sregs.u.s.ppc32.sr[i];

        }



        /* Sync BATs */

        for (i = 0; i < 8; i++) {

            env->DBAT[0][i] = sregs.u.s.ppc32.dbat[i] & 0xffffffff;

            env->DBAT[1][i] = sregs.u.s.ppc32.dbat[i] >> 32;

            env->IBAT[0][i] = sregs.u.s.ppc32.ibat[i] & 0xffffffff;

            env->IBAT[1][i] = sregs.u.s.ppc32.ibat[i] >> 32;

        }

    }

#endif



    return 0;

}
