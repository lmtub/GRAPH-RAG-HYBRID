void s390_sclp_extint(uint32_t parm)

{

    if (kvm_enabled()) {

        kvm_s390_service_interrupt(parm);

    } else {

        S390CPU *dummy_cpu = s390_cpu_addr2state(0);



        cpu_inject_ext(dummy_cpu, EXT_SERVICE, parm, 0);

    }

}
