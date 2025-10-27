void kvm_s390_interrupt(S390CPU *cpu, int type, uint32_t code)

{

    kvm_s390_interrupt_internal(cpu, type, code, 0, 0);

}
