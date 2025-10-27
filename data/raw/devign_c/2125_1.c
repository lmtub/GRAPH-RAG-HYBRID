static void spapr_machine_class_init(ObjectClass *oc, void *data)

{

    MachineClass *mc = MACHINE_CLASS(oc);

    sPAPRMachineClass *smc = SPAPR_MACHINE_CLASS(oc);

    FWPathProviderClass *fwc = FW_PATH_PROVIDER_CLASS(oc);

    NMIClass *nc = NMI_CLASS(oc);

    HotplugHandlerClass *hc = HOTPLUG_HANDLER_CLASS(oc);



    mc->desc = "pSeries Logical Partition (PAPR compliant)";



    /*

     * We set up the default / latest behaviour here.  The class_init

     * functions for the specific versioned machine types can override

     * these details for backwards compatibility

     */

    mc->init = ppc_spapr_init;

    mc->reset = ppc_spapr_reset;

    mc->block_default_type = IF_SCSI;

    mc->max_cpus = MAX_CPUMASK_BITS;

    mc->no_parallel = 1;

    mc->default_boot_order = "";

    mc->default_ram_size = 512 * M_BYTE;

    mc->kvm_type = spapr_kvm_type;

    mc->has_dynamic_sysbus = true;

    mc->pci_allow_0_address = true;

    mc->get_hotplug_handler = spapr_get_hotpug_handler;

    hc->pre_plug = spapr_machine_device_pre_plug;

    hc->plug = spapr_machine_device_plug;

    hc->unplug = spapr_machine_device_unplug;

    mc->cpu_index_to_socket_id = spapr_cpu_index_to_socket_id;

    mc->query_hotpluggable_cpus = spapr_query_hotpluggable_cpus;



    smc->dr_lmb_enabled = true;

    smc->dr_cpu_enabled = true;

    fwc->get_dev_path = spapr_get_fw_dev_path;

    nc->nmi_monitor_handler = spapr_nmi;

}
