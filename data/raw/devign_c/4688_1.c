static int kvm_set_ioeventfd_mmio(int fd, hwaddr addr, uint32_t val,

                                  bool assign, uint32_t size, bool datamatch)

{

    int ret;

    struct kvm_ioeventfd iofd;



    iofd.datamatch = datamatch ? adjust_ioeventfd_endianness(val, size) : 0;

    iofd.addr = addr;

    iofd.len = size;

    iofd.flags = 0;

    iofd.fd = fd;



    if (!kvm_enabled()) {

        return -ENOSYS;

    }



    if (datamatch) {

        iofd.flags |= KVM_IOEVENTFD_FLAG_DATAMATCH;

    }

    if (!assign) {

        iofd.flags |= KVM_IOEVENTFD_FLAG_DEASSIGN;

    }



    ret = kvm_vm_ioctl(kvm_state, KVM_IOEVENTFD, &iofd);



    if (ret < 0) {

        return -errno;

    }



    return 0;

}
