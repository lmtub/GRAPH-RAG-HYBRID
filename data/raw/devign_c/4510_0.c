int kvm_set_signal_mask(CPUState *cpu, const sigset_t *sigset)

{

    struct kvm_signal_mask *sigmask;

    int r;



    if (!sigset) {

        return kvm_vcpu_ioctl(cpu, KVM_SET_SIGNAL_MASK, NULL);

    }



    sigmask = g_malloc(sizeof(*sigmask) + sizeof(*sigset));



    sigmask->len = 8;

    memcpy(sigmask->sigset, sigset, sizeof(*sigset));

    r = kvm_vcpu_ioctl(cpu, KVM_SET_SIGNAL_MASK, sigmask);

    g_free(sigmask);



    return r;

}
