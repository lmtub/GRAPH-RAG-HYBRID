static void kvm_getput_reg(__u64 *kvm_reg, target_ulong *qemu_reg, int set)

{

    if (set)

        *kvm_reg = *qemu_reg;

    else

        *qemu_reg = *kvm_reg;

}
