void kvm_arm_reset_vcpu(ARMCPU *cpu)

{

    /* Feed the kernel back its initial register state */

    memmove(cpu->cpreg_values, cpu->cpreg_reset_values,

            cpu->cpreg_array_len * sizeof(cpu->cpreg_values[0]));



    if (!write_list_to_kvmstate(cpu)) {

        abort();

    }

}
