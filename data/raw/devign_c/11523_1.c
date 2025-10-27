static void kvmclock_realize(DeviceState *dev, Error **errp)
{
    KVMClockState *s = KVM_CLOCK(dev);
    kvm_update_clock(s);
    qemu_add_vm_change_state_handler(kvmclock_vm_state_change, s);