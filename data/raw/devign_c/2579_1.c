static void kvm_reset_vcpu(void *opaque)

{

    CPUState *env = opaque;



    kvm_arch_reset_vcpu(env);

    if (kvm_arch_put_registers(env)) {

        fprintf(stderr, "Fatal: kvm vcpu reset failed\n");

        abort();

    }

}
