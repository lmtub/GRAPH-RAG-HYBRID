static int kvm_init(MachineState *ms)

{

    MachineClass *mc = MACHINE_GET_CLASS(ms);

    static const char upgrade_note[] =

        "Please upgrade to at least kernel 2.6.29 or recent kvm-kmod\n"

        "(see http://sourceforge.net/projects/kvm).\n";

    struct {

        const char *name;

        int num;

    } num_cpus[] = {

        { "SMP",          smp_cpus },

        { "hotpluggable", max_cpus },

        { NULL, }

    }, *nc = num_cpus;

    int soft_vcpus_limit, hard_vcpus_limit;

    KVMState *s;

    const KVMCapabilityInfo *missing_cap;

    int ret;

    int i, type = 0;

    const char *kvm_type;



    s = KVM_STATE(ms->accelerator);



    /*

     * On systems where the kernel can support different base page

     * sizes, host page size may be different from TARGET_PAGE_SIZE,

     * even with KVM.  TARGET_PAGE_SIZE is assumed to be the minimum

     * page size for the system though.

     */

    assert(TARGET_PAGE_SIZE <= getpagesize());

    page_size_init();



    s->sigmask_len = 8;



#ifdef KVM_CAP_SET_GUEST_DEBUG

    QTAILQ_INIT(&s->kvm_sw_breakpoints);

#endif

    s->vmfd = -1;

    s->fd = qemu_open("/dev/kvm", O_RDWR);

    if (s->fd == -1) {

        fprintf(stderr, "Could not access KVM kernel module: %m\n");

        ret = -errno;

        goto err;

    }



    ret = kvm_ioctl(s, KVM_GET_API_VERSION, 0);

    if (ret < KVM_API_VERSION) {

        if (ret >= 0) {

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



    s->nr_slots = kvm_check_extension(s, KVM_CAP_NR_MEMSLOTS);



    /* If unspecified, use the default value */

    if (!s->nr_slots) {

        s->nr_slots = 32;

    }



    s->slots = g_malloc0(s->nr_slots * sizeof(KVMSlot));



    for (i = 0; i < s->nr_slots; i++) {

        s->slots[i].slot = i;

    }



    /* check the vcpu limits */

    soft_vcpus_limit = kvm_recommended_vcpus(s);

    hard_vcpus_limit = kvm_max_vcpus(s);



    while (nc->name) {

        if (nc->num > soft_vcpus_limit) {

            fprintf(stderr,

                    "Warning: Number of %s cpus requested (%d) exceeds "

                    "the recommended cpus supported by KVM (%d)\n",

                    nc->name, nc->num, soft_vcpus_limit);



            if (nc->num > hard_vcpus_limit) {

                fprintf(stderr, "Number of %s cpus requested (%d) exceeds "

                        "the maximum cpus supported by KVM (%d)\n",

                        nc->name, nc->num, hard_vcpus_limit);

                exit(1);

            }

        }

        nc++;

    }



    kvm_type = qemu_opt_get(qemu_get_machine_opts(), "kvm-type");

    if (mc->kvm_type) {

        type = mc->kvm_type(kvm_type);

    } else if (kvm_type) {

        ret = -EINVAL;

        fprintf(stderr, "Invalid argument kvm-type=%s\n", kvm_type);

        goto err;

    }



    do {

        ret = kvm_ioctl(s, KVM_CREATE_VM, type);

    } while (ret == -EINTR);



    if (ret < 0) {

        fprintf(stderr, "ioctl(KVM_CREATE_VM) failed: %d %s\n", -ret,

                strerror(-ret));



#ifdef TARGET_S390X

        fprintf(stderr, "Please add the 'switch_amode' kernel parameter to "

                        "your host kernel command line\n");

#endif

        goto err;

    }



    s->vmfd = ret;

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



#ifdef KVM_CAP_PIT_STATE2

    s->pit_state2 = kvm_check_extension(s, KVM_CAP_PIT_STATE2);

#endif



#ifdef KVM_CAP_IRQ_ROUTING

    s->direct_msi = (kvm_check_extension(s, KVM_CAP_SIGNAL_MSI) > 0);

#endif



    s->intx_set_mask = kvm_check_extension(s, KVM_CAP_PCI_2_3);



    s->irq_set_ioctl = KVM_IRQ_LINE;

    if (kvm_check_extension(s, KVM_CAP_IRQ_INJECT_STATUS)) {

        s->irq_set_ioctl = KVM_IRQ_LINE_STATUS;

    }



#ifdef KVM_CAP_READONLY_MEM

    kvm_readonly_mem_allowed =

        (kvm_check_extension(s, KVM_CAP_READONLY_MEM) > 0);

#endif



    kvm_eventfds_allowed =

        (kvm_check_extension(s, KVM_CAP_IOEVENTFD) > 0);



    kvm_irqfds_allowed =

        (kvm_check_extension(s, KVM_CAP_IRQFD) > 0);



    kvm_resamplefds_allowed =

        (kvm_check_extension(s, KVM_CAP_IRQFD_RESAMPLE) > 0);



    ret = kvm_arch_init(s);

    if (ret < 0) {

        goto err;

    }



    ret = kvm_irqchip_create(s);

    if (ret < 0) {

        goto err;

    }



    kvm_state = s;

    memory_listener_register(&kvm_memory_listener, &address_space_memory);

    memory_listener_register(&kvm_io_listener, &address_space_io);



    s->many_ioeventfds = kvm_check_many_ioeventfds();



    cpu_interrupt_handler = kvm_handle_interrupt;



    return 0;



err:

    assert(ret < 0);

    if (s->vmfd >= 0) {

        close(s->vmfd);

    }

    if (s->fd != -1) {

        close(s->fd);

    }

    g_free(s->slots);



    return ret;

}
