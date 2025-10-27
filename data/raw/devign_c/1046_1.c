static void kvmclock_pre_save(void *opaque)

{

    KVMClockState *s = opaque;

    struct kvm_clock_data data;

    int ret;



    if (s->clock_valid) {

        return;

    }

    ret = kvm_vm_ioctl(kvm_state, KVM_GET_CLOCK, &data);

    if (ret < 0) {

        fprintf(stderr, "KVM_GET_CLOCK failed: %s\n", strerror(ret));

        data.clock = 0;

    }

    s->clock = data.clock;

    /*

     * If the VM is stopped, declare the clock state valid to avoid re-reading

     * it on next vmsave (which would return a different value). Will be reset

     * when the VM is continued.

     */

    s->clock_valid = !runstate_is_running();

}
