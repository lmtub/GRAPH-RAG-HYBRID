void pc_machine_done(Notifier *notifier, void *data)

{

    PCMachineState *pcms = container_of(notifier,

                                        PCMachineState, machine_done);

    PCIBus *bus = pcms->bus;



    /* set the number of CPUs */

    rtc_set_cpus_count(pcms->rtc, pcms->boot_cpus);



    if (bus) {

        int extra_hosts = 0;



        QLIST_FOREACH(bus, &bus->child, sibling) {

            /* look for expander root buses */

            if (pci_bus_is_root(bus)) {

                extra_hosts++;

            }

        }

        if (extra_hosts && pcms->fw_cfg) {

            uint64_t *val = g_malloc(sizeof(*val));

            *val = cpu_to_le64(extra_hosts);

            fw_cfg_add_file(pcms->fw_cfg,

                    "etc/extra-pci-roots", val, sizeof(*val));

        }

    }



    acpi_setup();

    if (pcms->fw_cfg) {

        pc_build_smbios(pcms);

        pc_build_feature_control_file(pcms);

        /* update FW_CFG_NB_CPUS to account for -device added CPUs */

        fw_cfg_modify_i16(pcms->fw_cfg, FW_CFG_NB_CPUS, pcms->boot_cpus);

    }



    if (pcms->apic_id_limit > 255) {

        IntelIOMMUState *iommu = INTEL_IOMMU_DEVICE(x86_iommu_get_default());



        if (!iommu || !iommu->x86_iommu.intr_supported ||

            iommu->intr_eim != ON_OFF_AUTO_ON) {

            error_report("current -smp configuration requires "

                         "Extended Interrupt Mode enabled. "

                         "You can add an IOMMU using: "

                         "-device intel-iommu,intremap=on,eim=on");

            exit(EXIT_FAILURE);

        }

    }

}
