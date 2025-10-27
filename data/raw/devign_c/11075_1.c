int kvm_init(void)

{

    static const char upgrade_note[] =

        "Please upgrade to at least kernel 2.6.29 or recent kvm-kmod\n"

        "(see http://sourceforge.net/projects/kvm).\n";

    KVMState *s;

    const KVMCapabilityInfo *missing_cap;

    int ret;

    int i;



    s = g_malloc0(sizeof(KVMState));



#ifdef KVM_CAP_SET_GUEST_DEBUG

    QTAILQ_INIT(&s->kvm_sw_breakpoints);

#endif

    for (i = 0; i < ARRAY_SIZE(s->slots); i++) {

        s->slots[i].slot = i;

    }

    s->vmfd = -1;

    s->fd = qemu_open("/dev/kvm", O_RDWR);

    if (s->fd == -1) {

        fprintf(stderr, "Could not access KVM kernel module: %m\n");

        ret = -errno;

        goto err;

    }



    ret = kvm_ioctl(s, KVM_GET_API_VERSION, 0);

    if (ret < KVM_API_VERSION) {

        if (ret > 0) {

            ret = -EINVAL;

        }

        fprintf(stderr, "kvm version too old\n");

        goto err;

    }



    if (ret > KVM_API_VERSION) {

        ret = -EINVAL;

        fprintf(stderr, "kvm version not supported\n");

        goto err;

    }



    s->vmfd = kvm_ioctl(s, KVM_CREATE_VM, 0);

    if (s->vmfd < 0) {

#ifdef TARGET_S390X

        fprintf(stderr, "Please add the 'switch_amode' kernel parameter to "

                        "your host kernel command line\n");

#endif

        ret = s->vmfd;

        goto err;

    }



    missing_cap = kvm_check_extension_list(s, kvm_required_capabilites);

    if (!missing_cap) {

        missing_cap =

            kvm_check_extension_list(s, kvm_arch_required_capabilities);

    }

    if (missing_cap) {

        ret = -EINVAL;

        fprintf(stderr, "kvm does not support %s\n%s",

                missing_cap->name, upgrade_note);

        goto err;

    }



    s->coalesced_mmio = kvm_check_extension(s, KVM_CAP_COALESCED_MMIO);



    s->broken_set_mem_region = 1;

    ret = kvm_check_extension(s, KVM_CAP_JOIN_MEMORY_REGIONS_WORKS);

    if (ret > 0) {

        s->broken_set_mem_region = 0;

    }



#ifdef KVM_CAP_VCPU_EVENTS

    s->vcpu_events = kvm_check_extension(s, KVM_CAP_VCPU_EVENTS);

#endif



    s->robust_singlestep =

        kvm_check_extension(s, KVM_CAP_X86_ROBUST_SINGLESTEP);



#ifdef KVM_CAP_DEBUGREGS

    s->debugregs = kvm_check_extension(s, KVM_CAP_DEBUGREGS);

#endif



#ifdef KVM_CAP_XSAVE

    s->xsave = kvm_check_extension(s, KVM_CAP_XSAVE);

#endif



#ifdef KVM_CAP_XCRS

    s->xcrs = kvm_check_extension(s, KVM_CAP_XCRS);

#endif



    ret = kvm_arch_init(s);

    if (ret < 0) {

        goto err;

    }



    kvm_state = s;

    cpu_register_phys_memory_client(&kvm_cpu_phys_memory_client);



    s->many_ioeventfds = kvm_check_many_ioeventfds();



    cpu_interrupt_handler = kvm_handle_interrupt;



    return 0;



err:

    if (s) {

        if (s->vmfd >= 0) {

            close(s->vmfd);

        }

        if (s->fd != -1) {

            close(s->fd);

        }

    }

    g_free(s);



    return ret;

}
