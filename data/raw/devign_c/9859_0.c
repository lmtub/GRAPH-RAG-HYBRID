void kvm_s390_interrupt_internal(S390CPU *cpu, int type, uint32_t parm,

                                 uint64_t parm64, int vm)

{

    CPUState *cs = CPU(cpu);

    struct kvm_s390_interrupt kvmint;

    int r;



    if (!cs->kvm_state) {

        return;

    }



    kvmint.type = type;

    kvmint.parm = parm;

    kvmint.parm64 = parm64;



    if (vm) {

        r = kvm_vm_ioctl(cs->kvm_state, KVM_S390_INTERRUPT, &kvmint);

    } else {

        r = kvm_vcpu_ioctl(cs, KVM_S390_INTERRUPT, &kvmint);

    }



    if (r < 0) {

        fprintf(stderr, "KVM failed to inject interrupt\n");

        exit(1);

    }

}
