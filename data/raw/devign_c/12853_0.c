int kvm_arch_put_registers(CPUState *cs, int level)

{

    S390CPU *cpu = S390_CPU(cs);

    CPUS390XState *env = &cpu->env;

    struct kvm_sregs sregs;

    struct kvm_regs regs;

    struct kvm_fpu fpu;

    int r;

    int i;



    /* always save the PSW  and the GPRS*/

    cs->kvm_run->psw_addr = env->psw.addr;

    cs->kvm_run->psw_mask = env->psw.mask;



    if (cap_sync_regs && cs->kvm_run->kvm_valid_regs & KVM_SYNC_GPRS) {

        for (i = 0; i < 16; i++) {

            cs->kvm_run->s.regs.gprs[i] = env->regs[i];

            cs->kvm_run->kvm_dirty_regs |= KVM_SYNC_GPRS;

        }

    } else {

        for (i = 0; i < 16; i++) {

            regs.gprs[i] = env->regs[i];

        }

        r = kvm_vcpu_ioctl(cs, KVM_SET_REGS, &regs);

        if (r < 0) {

            return r;

        }

    }



    /* Floating point */

    for (i = 0; i < 16; i++) {

        fpu.fprs[i] = env->fregs[i].ll;

    }

    fpu.fpc = env->fpc;



    r = kvm_vcpu_ioctl(cs, KVM_SET_FPU, &fpu);

    if (r < 0) {

        return r;

    }



    /* Do we need to save more than that? */

    if (level == KVM_PUT_RUNTIME_STATE) {

        return 0;

    }



    /*

     * These ONE_REGS are not protected by a capability. As they are only

     * necessary for migration we just trace a possible error, but don't

     * return with an error return code.

     */

    kvm_set_one_reg(cs, KVM_REG_S390_CPU_TIMER, &env->cputm);

    kvm_set_one_reg(cs, KVM_REG_S390_CLOCK_COMP, &env->ckc);

    kvm_set_one_reg(cs, KVM_REG_S390_TODPR, &env->todpr);

    kvm_set_one_reg(cs, KVM_REG_S390_GBEA, &env->gbea);

    kvm_set_one_reg(cs, KVM_REG_S390_PP, &env->pp);



    if (cap_async_pf) {

        r = kvm_set_one_reg(cs, KVM_REG_S390_PFTOKEN, &env->pfault_token);

        if (r < 0) {

            return r;

        }

        r = kvm_set_one_reg(cs, KVM_REG_S390_PFCOMPARE, &env->pfault_compare);

        if (r < 0) {

            return r;

        }

        r = kvm_set_one_reg(cs, KVM_REG_S390_PFSELECT, &env->pfault_select);

        if (r < 0) {

            return r;

        }

    }



    if (cap_sync_regs &&

        cs->kvm_run->kvm_valid_regs & KVM_SYNC_ACRS &&

        cs->kvm_run->kvm_valid_regs & KVM_SYNC_CRS) {

        for (i = 0; i < 16; i++) {

            cs->kvm_run->s.regs.acrs[i] = env->aregs[i];

            cs->kvm_run->s.regs.crs[i] = env->cregs[i];

        }

        cs->kvm_run->kvm_dirty_regs |= KVM_SYNC_ACRS;

        cs->kvm_run->kvm_dirty_regs |= KVM_SYNC_CRS;

    } else {

        for (i = 0; i < 16; i++) {

            sregs.acrs[i] = env->aregs[i];

            sregs.crs[i] = env->cregs[i];

        }

        r = kvm_vcpu_ioctl(cs, KVM_SET_SREGS, &sregs);

        if (r < 0) {

            return r;

        }

    }



    /* Finally the prefix */

    if (cap_sync_regs && cs->kvm_run->kvm_valid_regs & KVM_SYNC_PREFIX) {

        cs->kvm_run->s.regs.prefix = env->psa;

        cs->kvm_run->kvm_dirty_regs |= KVM_SYNC_PREFIX;

    } else {

        /* prefix is only supported via sync regs */

    }

    return 0;

}
