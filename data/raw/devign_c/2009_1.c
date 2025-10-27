void s390_machine_reset(void)

{

    S390CPU *ipl_cpu = S390_CPU(qemu_get_cpu(0));



    qemu_devices_reset();

    s390_cmma_reset();

    s390_crypto_reset();



    /* all cpus are stopped - configure and start the ipl cpu only */

    s390_ipl_prepare_cpu(ipl_cpu);

    s390_cpu_set_state(CPU_STATE_OPERATING, ipl_cpu);

}
