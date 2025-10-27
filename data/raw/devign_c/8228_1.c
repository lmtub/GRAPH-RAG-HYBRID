int kvm_arch_get_registers(CPUState *cs)

{

    PowerPCCPU *cpu = POWERPC_CPU(cs);

    CPUPPCState *env = &cpu->env;

    struct kvm_regs regs;

    struct kvm_sregs sregs;

    uint32_t cr;

    int i, ret;



    ret = kvm_vcpu_ioctl(cs, KVM_GET_REGS, &regs);

    if (ret < 0)

        return ret;



    cr = regs.cr;

    for (i = 7; i >= 0; i--) {

        env->crf[i] = cr & 15;

        cr >>= 4;

    }



    env->ctr = regs.ctr;

    env->lr = regs.lr;

    cpu_write_xer(env, regs.xer);

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



    env->spr[SPR_BOOKE_PID] = regs.pid;



    for (i = 0;i < 32; i++)

        env->gpr[i] = regs.gpr[i];



    kvm_get_fp(cs);



    if (cap_booke_sregs) {

        ret = kvm_vcpu_ioctl(cs, KVM_GET_SREGS, &sregs);

        if (ret < 0) {

            return ret;

        }



        if (sregs.u.e.features & KVM_SREGS_E_BASE) {

            env->spr[SPR_BOOKE_CSRR0] = sregs.u.e.csrr0;

            env->spr[SPR_BOOKE_CSRR1] = sregs.u.e.csrr1;

            env->spr[SPR_BOOKE_ESR] = sregs.u.e.esr;

            env->spr[SPR_BOOKE_DEAR] = sregs.u.e.dear;

            env->spr[SPR_BOOKE_MCSR] = sregs.u.e.mcsr;

            env->spr[SPR_BOOKE_TSR] = sregs.u.e.tsr;

            env->spr[SPR_BOOKE_TCR] = sregs.u.e.tcr;

            env->spr[SPR_DECR] = sregs.u.e.dec;

            env->spr[SPR_TBL] = sregs.u.e.tb & 0xffffffff;

            env->spr[SPR_TBU] = sregs.u.e.tb >> 32;

            env->spr[SPR_VRSAVE] = sregs.u.e.vrsave;

        }



        if (sregs.u.e.features & KVM_SREGS_E_ARCH206) {

            env->spr[SPR_BOOKE_PIR] = sregs.u.e.pir;

            env->spr[SPR_BOOKE_MCSRR0] = sregs.u.e.mcsrr0;

            env->spr[SPR_BOOKE_MCSRR1] = sregs.u.e.mcsrr1;

            env->spr[SPR_BOOKE_DECAR] = sregs.u.e.decar;

            env->spr[SPR_BOOKE_IVPR] = sregs.u.e.ivpr;

        }



        if (sregs.u.e.features & KVM_SREGS_E_64) {

            env->spr[SPR_BOOKE_EPCR] = sregs.u.e.epcr;

        }



        if (sregs.u.e.features & KVM_SREGS_E_SPRG8) {

            env->spr[SPR_BOOKE_SPRG8] = sregs.u.e.sprg8;

        }



        if (sregs.u.e.features & KVM_SREGS_E_IVOR) {

            env->spr[SPR_BOOKE_IVOR0] = sregs.u.e.ivor_low[0];

            env->spr[SPR_BOOKE_IVOR1] = sregs.u.e.ivor_low[1];

            env->spr[SPR_BOOKE_IVOR2] = sregs.u.e.ivor_low[2];

            env->spr[SPR_BOOKE_IVOR3] = sregs.u.e.ivor_low[3];

            env->spr[SPR_BOOKE_IVOR4] = sregs.u.e.ivor_low[4];

            env->spr[SPR_BOOKE_IVOR5] = sregs.u.e.ivor_low[5];

            env->spr[SPR_BOOKE_IVOR6] = sregs.u.e.ivor_low[6];

            env->spr[SPR_BOOKE_IVOR7] = sregs.u.e.ivor_low[7];

            env->spr[SPR_BOOKE_IVOR8] = sregs.u.e.ivor_low[8];

            env->spr[SPR_BOOKE_IVOR9] = sregs.u.e.ivor_low[9];

            env->spr[SPR_BOOKE_IVOR10] = sregs.u.e.ivor_low[10];

            env->spr[SPR_BOOKE_IVOR11] = sregs.u.e.ivor_low[11];

            env->spr[SPR_BOOKE_IVOR12] = sregs.u.e.ivor_low[12];

            env->spr[SPR_BOOKE_IVOR13] = sregs.u.e.ivor_low[13];

            env->spr[SPR_BOOKE_IVOR14] = sregs.u.e.ivor_low[14];

            env->spr[SPR_BOOKE_IVOR15] = sregs.u.e.ivor_low[15];



            if (sregs.u.e.features & KVM_SREGS_E_SPE) {

                env->spr[SPR_BOOKE_IVOR32] = sregs.u.e.ivor_high[0];

                env->spr[SPR_BOOKE_IVOR33] = sregs.u.e.ivor_high[1];

                env->spr[SPR_BOOKE_IVOR34] = sregs.u.e.ivor_high[2];

            }



            if (sregs.u.e.features & KVM_SREGS_E_PM) {

                env->spr[SPR_BOOKE_IVOR35] = sregs.u.e.ivor_high[3];

            }



            if (sregs.u.e.features & KVM_SREGS_E_PC) {

                env->spr[SPR_BOOKE_IVOR36] = sregs.u.e.ivor_high[4];

                env->spr[SPR_BOOKE_IVOR37] = sregs.u.e.ivor_high[5];

            }

        }



        if (sregs.u.e.features & KVM_SREGS_E_ARCH206_MMU) {

            env->spr[SPR_BOOKE_MAS0] = sregs.u.e.mas0;

            env->spr[SPR_BOOKE_MAS1] = sregs.u.e.mas1;

            env->spr[SPR_BOOKE_MAS2] = sregs.u.e.mas2;

            env->spr[SPR_BOOKE_MAS3] = sregs.u.e.mas7_3 & 0xffffffff;

            env->spr[SPR_BOOKE_MAS4] = sregs.u.e.mas4;

            env->spr[SPR_BOOKE_MAS6] = sregs.u.e.mas6;

            env->spr[SPR_BOOKE_MAS7] = sregs.u.e.mas7_3 >> 32;

            env->spr[SPR_MMUCFG] = sregs.u.e.mmucfg;

            env->spr[SPR_BOOKE_TLB0CFG] = sregs.u.e.tlbcfg[0];

            env->spr[SPR_BOOKE_TLB1CFG] = sregs.u.e.tlbcfg[1];

        }



        if (sregs.u.e.features & KVM_SREGS_EXP) {

            env->spr[SPR_BOOKE_EPR] = sregs.u.e.epr;

        }



        if (sregs.u.e.features & KVM_SREGS_E_PD) {

            env->spr[SPR_BOOKE_EPLC] = sregs.u.e.eplc;

            env->spr[SPR_BOOKE_EPSC] = sregs.u.e.epsc;

        }



        if (sregs.u.e.impl_id == KVM_SREGS_E_IMPL_FSL) {

            env->spr[SPR_E500_SVR] = sregs.u.e.impl.fsl.svr;

            env->spr[SPR_Exxx_MCAR] = sregs.u.e.impl.fsl.mcar;

            env->spr[SPR_HID0] = sregs.u.e.impl.fsl.hid0;



            if (sregs.u.e.impl.fsl.features & KVM_SREGS_E_FSL_PIDn) {

                env->spr[SPR_BOOKE_PID1] = sregs.u.e.impl.fsl.pid1;

                env->spr[SPR_BOOKE_PID2] = sregs.u.e.impl.fsl.pid2;

            }

        }

    }



    if (cap_segstate) {

        ret = kvm_vcpu_ioctl(cs, KVM_GET_SREGS, &sregs);

        if (ret < 0) {

            return ret;

        }



        ppc_store_sdr1(env, sregs.u.s.sdr1);



        /* Sync SLB */

#ifdef TARGET_PPC64

        /*

         * The packed SLB array we get from KVM_GET_SREGS only contains

         * information about valid entries. So we flush our internal

         * copy to get rid of stale ones, then put all valid SLB entries

         * back in.

         */

        memset(env->slb, 0, sizeof(env->slb));

        for (i = 0; i < ARRAY_SIZE(env->slb); i++) {

            target_ulong rb = sregs.u.s.ppc64.slb[i].slbe;

            target_ulong rs = sregs.u.s.ppc64.slb[i].slbv;

            /*

             * Only restore valid entries

             */

            if (rb & SLB_ESID_V) {

                ppc_store_slb(env, rb, rs);

            }

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



    if (cap_hior) {

        kvm_get_one_spr(cs, KVM_REG_PPC_HIOR, SPR_HIOR);

    }



    if (cap_one_reg) {

        int i;



        /* We deliberately ignore errors here, for kernels which have

         * the ONE_REG calls, but don't support the specific

         * registers, there's a reasonable chance things will still

         * work, at least until we try to migrate. */

        for (i = 0; i < 1024; i++) {

            uint64_t id = env->spr_cb[i].one_reg_id;



            if (id != 0) {

                kvm_get_one_spr(cs, id, i);

            }

        }



#ifdef TARGET_PPC64

        if (cap_papr) {

            if (kvm_get_vpa(cs) < 0) {

                DPRINTF("Warning: Unable to get VPA information from KVM\n");

            }

        }

#endif

    }



    return 0;

}
