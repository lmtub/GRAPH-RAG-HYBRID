static int kvm_get_xsave(CPUState *env)

{

#ifdef KVM_CAP_XSAVE

    struct kvm_xsave* xsave;

    int ret, i;

    uint16_t cwd, swd, twd, fop;



    if (!kvm_has_xsave()) {

        return kvm_get_fpu(env);

    }



    xsave = qemu_memalign(4096, sizeof(struct kvm_xsave));

    ret = kvm_vcpu_ioctl(env, KVM_GET_XSAVE, xsave);

    if (ret < 0) {

        qemu_free(xsave);

        return ret;

    }



    cwd = (uint16_t)xsave->region[0];

    swd = (uint16_t)(xsave->region[0] >> 16);

    twd = (uint16_t)xsave->region[1];

    fop = (uint16_t)(xsave->region[1] >> 16);

    env->fpstt = (swd >> 11) & 7;

    env->fpus = swd;

    env->fpuc = cwd;

    for (i = 0; i < 8; ++i) {

        env->fptags[i] = !((twd >> i) & 1);

    }

    env->mxcsr = xsave->region[XSAVE_MXCSR];

    memcpy(env->fpregs, &xsave->region[XSAVE_ST_SPACE],

            sizeof env->fpregs);

    memcpy(env->xmm_regs, &xsave->region[XSAVE_XMM_SPACE],

            sizeof env->xmm_regs);

    env->xstate_bv = *(uint64_t *)&xsave->region[XSAVE_XSTATE_BV];

    memcpy(env->ymmh_regs, &xsave->region[XSAVE_YMMH_SPACE],

            sizeof env->ymmh_regs);

    qemu_free(xsave);

    return 0;

#else

    return kvm_get_fpu(env);

#endif

}
