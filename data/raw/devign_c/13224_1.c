int kvm_init_vcpu(CPUState *env)

{

    KVMState *s = kvm_state;

    long mmap_size;

    int ret;



    DPRINTF("kvm_init_vcpu\n");



    ret = kvm_vm_ioctl(s, KVM_CREATE_VCPU, env->cpu_index);

    if (ret < 0) {

        DPRINTF("kvm_create_vcpu failed\n");

        goto err;

    }



    env->kvm_fd = ret;

    env->kvm_state = s;



    mmap_size = kvm_ioctl(s, KVM_GET_VCPU_MMAP_SIZE, 0);

    if (mmap_size < 0) {

        DPRINTF("KVM_GET_VCPU_MMAP_SIZE failed\n");

        goto err;

    }



    env->kvm_run = mmap(NULL, mmap_size, PROT_READ | PROT_WRITE, MAP_SHARED,

                        env->kvm_fd, 0);

    if (env->kvm_run == MAP_FAILED) {

        ret = -errno;

        DPRINTF("mmap'ing vcpu state failed\n");

        goto err;

    }



#ifdef KVM_CAP_COALESCED_MMIO

    if (s->coalesced_mmio && !s->coalesced_mmio_ring) {

        s->coalesced_mmio_ring =

            (void *)env->kvm_run + s->coalesced_mmio * PAGE_SIZE;

    }

#endif



    ret = kvm_arch_init_vcpu(env);

    if (ret == 0) {

        qemu_register_reset(kvm_reset_vcpu, env);

        kvm_arch_reset_vcpu(env);

    }

err:

    return ret;

}
