int kvm_init(int smp_cpus)
{
    KVMState *s;
    int ret;
    int i;
    if (smp_cpus > 1)
        return -EINVAL;
    s = qemu_mallocz(sizeof(KVMState));
    if (s == NULL)
        return -ENOMEM;
    for (i = 0; i < ARRAY_SIZE(s->slots); i++)
        s->slots[i].slot = i;
    s->vmfd = -1;
    s->fd = open("/dev/kvm", O_RDWR);
    if (s->fd == -1) {
        fprintf(stderr, "Could not access KVM kernel module: %m\n");
        ret = -errno;
        goto err;
    }
    ret = kvm_ioctl(s, KVM_GET_API_VERSION, 0);
    if (ret < KVM_API_VERSION) {
            ret = -EINVAL;
        fprintf(stderr, "kvm version too old\n");
        goto err;
    }
    if (ret > KVM_API_VERSION) {
        ret = -EINVAL;
        fprintf(stderr, "kvm version not supported\n");
        goto err;
    }
    s->vmfd = kvm_ioctl(s, KVM_CREATE_VM, 0);
    if (s->vmfd < 0)
        goto err;
    /* initially, KVM allocated its own memory and we had to jump through
     * hooks to make phys_ram_base point to this.  Modern versions of KVM
     * just use a user allocated buffer so we can use phys_ram_base
     * unmodified.  Make sure we have a sufficiently modern version of KVM.
     */
    ret = kvm_ioctl(s, KVM_CHECK_EXTENSION, KVM_CAP_USER_MEMORY);
    if (ret <= 0) {
        if (ret == 0)
            ret = -EINVAL;
        fprintf(stderr, "kvm does not support KVM_CAP_USER_MEMORY\n");
        goto err;
    }
    /* There was a nasty bug in < kvm-80 that prevents memory slots from being
     * destroyed properly.  Since we rely on this capability, refuse to work
     * with any kernel without this capability. */
    ret = kvm_ioctl(s, KVM_CHECK_EXTENSION,
                    KVM_CAP_DESTROY_MEMORY_REGION_WORKS);
    if (ret <= 0) {
        if (ret == 0)
            ret = -EINVAL;
        fprintf(stderr,
                "KVM kernel module broken (DESTROY_MEMORY_REGION)\n"
                "Please upgrade to at least kvm-81.\n");
        goto err;
    }
    ret = kvm_arch_init(s, smp_cpus);
    if (ret < 0)
        goto err;
    kvm_state = s;
    return 0;
err:
    if (s) {
        if (s->vmfd != -1)
            close(s->vmfd);
        if (s->fd != -1)
            close(s->fd);
    }
    qemu_free(s);
    return ret;
}