static void virt_machine_class_init(ObjectClass *oc, void *data)

{

    MachineClass *mc = MACHINE_CLASS(oc);



    mc->init = machvirt_init;

    /* Start max_cpus at the maximum QEMU supports. We'll further restrict

     * it later in machvirt_init, where we have more information about the

     * configuration of the particular instance.

     */

    mc->max_cpus = MAX_CPUMASK_BITS;

    mc->has_dynamic_sysbus = true;

    mc->block_default_type = IF_VIRTIO;

    mc->no_cdrom = 1;

    mc->pci_allow_0_address = true;

}
