void kvm_s390_service_interrupt(S390CPU *cpu, uint32_t parm)

{

    kvm_s390_interrupt_internal(cpu, KVM_S390_INT_SERVICE, parm, 0 , 1);

}
