static int kvm_s390_io_adapter_map(S390FLICState *fs, uint32_t id,

                                   uint64_t map_addr, bool do_map)

{

    struct kvm_s390_io_adapter_req req = {

        .id = id,

        .type = do_map ? KVM_S390_IO_ADAPTER_MAP : KVM_S390_IO_ADAPTER_UNMAP,

        .addr = map_addr,

    };

    struct kvm_device_attr attr = {

        .group = KVM_DEV_FLIC_ADAPTER_MODIFY,

        .addr = (uint64_t)&req,

    };

    KVMS390FLICState *flic = KVM_S390_FLIC(fs);

    int r;



    if (!kvm_check_extension(kvm_state, KVM_CAP_IRQ_ROUTING)) {

        /* nothing to do */

        return 0;

    }



    r = ioctl(flic->fd, KVM_SET_DEVICE_ATTR, &attr);

    return r ? -errno : 0;

}
