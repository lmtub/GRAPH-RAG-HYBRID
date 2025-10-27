static int kvm_get_xsave(X86CPU *cpu)

{

    CPUX86State *env = &cpu->env;

    struct kvm_xsave* xsave = env->kvm_xsave_buf;

    int ret, i;

    const uint8_t *xmm, *ymmh, *zmmh;

    uint16_t cwd, swd, twd;



    if (!kvm_has_xsave()) {

        return kvm_get_fpu(cpu);

    }



    ret = kvm_vcpu_ioctl(CPU(cpu), KVM_GET_XSAVE, xsave);

    if (ret < 0) {

        return ret;

    }



    cwd = (uint16_t)xsave->region[XSAVE_FCW_FSW];

    swd = (uint16_t)(xsave->region[XSAVE_FCW_FSW] >> 16);

    twd = (uint16_t)xsave->region[XSAVE_FTW_FOP];

    env->fpop = (uint16_t)(xsave->region[XSAVE_FTW_FOP] >> 16);

    env->fpstt = (swd >> 11) & 7;

    env->fpus = swd;

    env->fpuc = cwd;

    for (i = 0; i < 8; ++i) {

        env->fptags[i] = !((twd >> i) & 1);

    }

    memcpy(&env->fpip, &xsave->region[XSAVE_CWD_RIP], sizeof(env->fpip));

    memcpy(&env->fpdp, &xsave->region[XSAVE_CWD_RDP], sizeof(env->fpdp));

    env->mxcsr = xsave->region[XSAVE_MXCSR];

    memcpy(env->fpregs, &xsave->region[XSAVE_ST_SPACE],

            sizeof env->fpregs);

    env->xstate_bv = *(uint64_t *)&xsave->region[XSAVE_XSTATE_BV];

    memcpy(env->bnd_regs, &xsave->region[XSAVE_BNDREGS],

            sizeof env->bnd_regs);

    memcpy(&env->bndcs_regs, &xsave->region[XSAVE_BNDCSR],

            sizeof(env->bndcs_regs));

    memcpy(env->opmask_regs, &xsave->region[XSAVE_OPMASK],

            sizeof env->opmask_regs);



    xmm = (const uint8_t *)&xsave->region[XSAVE_XMM_SPACE];

    ymmh = (const uint8_t *)&xsave->region[XSAVE_YMMH_SPACE];

    zmmh = (const uint8_t *)&xsave->region[XSAVE_ZMM_Hi256];

    for (i = 0; i < CPU_NB_REGS; i++, xmm += 16, ymmh += 16, zmmh += 32) {

        env->xmm_regs[i].XMM_Q(0) = ldq_p(xmm);

        env->xmm_regs[i].XMM_Q(1) = ldq_p(xmm+8);

        env->xmm_regs[i].XMM_Q(2) = ldq_p(ymmh);

        env->xmm_regs[i].XMM_Q(3) = ldq_p(ymmh+8);

        env->xmm_regs[i].XMM_Q(4) = ldq_p(zmmh);

        env->xmm_regs[i].XMM_Q(5) = ldq_p(zmmh+8);

        env->xmm_regs[i].XMM_Q(6) = ldq_p(zmmh+16);

        env->xmm_regs[i].XMM_Q(7) = ldq_p(zmmh+24);

    }



#ifdef TARGET_X86_64

    memcpy(&env->xmm_regs[16], &xsave->region[XSAVE_Hi16_ZMM],

           16 * sizeof env->xmm_regs[16]);

#endif

    return 0;

}
