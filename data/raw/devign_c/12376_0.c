static void virtio_notify(struct subchannel_id schid)

{

    kvm_hypercall(KVM_S390_VIRTIO_CCW_NOTIFY, *(u32*)&schid, 0);

}
