int s390_cpu_restart(S390CPU *cpu)

{

    if (kvm_enabled()) {

        return kvm_s390_cpu_restart(cpu);

    }

    return -ENOSYS;

}
