static int kvm_put_xsave(CPUState *env)

{

#ifdef KVM_CAP_XSAVE

    int i;

    struct kvm_xsave* xsave;

    uint16_t cwd, swd, twd, fop;



    if (!kvm_has_xsave())

        return kvm_put_fpu(env);



    xsave = qemu_memalign(4096, sizeof(struct kvm_xsave));

    memset(xsave, 0, sizeof(struct kvm_xsave));

    cwd = swd = twd = fop = 0;

    swd = env->fpus & ~(7 << 11);

    swd |= (env->fpstt & 7) << 11;

    cwd = env->fpuc;

    for (i = 0; i < 8; ++i)

        twd |= (!env->fptags[i]) << i;

    xsave->region[0] = (uint32_t)(swd << 16) + cwd;

    xsave->region[1] = (uint32_t)(fop << 16) + twd;

    memcpy(&xsave->region[XSAVE_ST_SPACE], env->fpregs,

            sizeof env->fpregs);

    memcpy(&xsave->region[XSAVE_XMM_SPACE], env->xmm_regs,

            sizeof env->xmm_regs);

    xsave->region[XSAVE_MXCSR] = env->mxcsr;

    *(uint64_t *)&xsave->region[XSAVE_XSTATE_BV] = env->xstate_bv;

    memcpy(&xsave->region[XSAVE_YMMH_SPACE], env->ymmh_regs,

            sizeof env->ymmh_regs);

    return kvm_vcpu_ioctl(env, KVM_SET_XSAVE, xsave);

#else

    return kvm_put_fpu(env);

#endif

}
