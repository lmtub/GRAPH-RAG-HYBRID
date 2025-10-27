static int kvm_put_fpu(CPUState *env)

{

    struct kvm_fpu fpu;

    int i;



    memset(&fpu, 0, sizeof fpu);

    fpu.fsw = env->fpus & ~(7 << 11);

    fpu.fsw |= (env->fpstt & 7) << 11;

    fpu.fcw = env->fpuc;

    for (i = 0; i < 8; ++i)

	fpu.ftwx |= (!env->fptags[i]) << i;

    memcpy(fpu.fpr, env->fpregs, sizeof env->fpregs);

    memcpy(fpu.xmm, env->xmm_regs, sizeof env->xmm_regs);

    fpu.mxcsr = env->mxcsr;



    return kvm_vcpu_ioctl(env, KVM_SET_FPU, &fpu);

}
