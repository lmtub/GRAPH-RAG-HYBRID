static FWCfgState *bochs_bios_init(AddressSpace *as, PCMachineState *pcms)

{

    FWCfgState *fw_cfg;

    uint64_t *numa_fw_cfg;

    int i;

    const CPUArchIdList *cpus;

    MachineClass *mc = MACHINE_GET_CLASS(pcms);



    fw_cfg = fw_cfg_init_io_dma(FW_CFG_IO_BASE, FW_CFG_IO_BASE + 4, as);

    fw_cfg_add_i16(fw_cfg, FW_CFG_NB_CPUS, pcms->boot_cpus);



    /* FW_CFG_MAX_CPUS is a bit confusing/problematic on x86:

     *

     * For machine types prior to 1.8, SeaBIOS needs FW_CFG_MAX_CPUS for

     * building MPTable, ACPI MADT, ACPI CPU hotplug and ACPI SRAT table,

     * that tables are based on xAPIC ID and QEMU<->SeaBIOS interface

     * for CPU hotplug also uses APIC ID and not "CPU index".

     * This means that FW_CFG_MAX_CPUS is not the "maximum number of CPUs",

     * but the "limit to the APIC ID values SeaBIOS may see".

     *

     * So for compatibility reasons with old BIOSes we are stuck with

     * "etc/max-cpus" actually being apic_id_limit

     */

    fw_cfg_add_i16(fw_cfg, FW_CFG_MAX_CPUS, (uint16_t)pcms->apic_id_limit);

    fw_cfg_add_i64(fw_cfg, FW_CFG_RAM_SIZE, (uint64_t)ram_size);

    fw_cfg_add_bytes(fw_cfg, FW_CFG_ACPI_TABLES,

                     acpi_tables, acpi_tables_len);

    fw_cfg_add_i32(fw_cfg, FW_CFG_IRQ0_OVERRIDE, kvm_allows_irq0_override());



    fw_cfg_add_bytes(fw_cfg, FW_CFG_E820_TABLE,

                     &e820_reserve, sizeof(e820_reserve));

    fw_cfg_add_file(fw_cfg, "etc/e820", e820_table,

                    sizeof(struct e820_entry) * e820_entries);



    fw_cfg_add_bytes(fw_cfg, FW_CFG_HPET, &hpet_cfg, sizeof(hpet_cfg));

    /* allocate memory for the NUMA channel: one (64bit) word for the number

     * of nodes, one word for each VCPU->node and one word for each node to

     * hold the amount of memory.

     */

    numa_fw_cfg = g_new0(uint64_t, 1 + pcms->apic_id_limit + nb_numa_nodes);

    numa_fw_cfg[0] = cpu_to_le64(nb_numa_nodes);

    cpus = mc->possible_cpu_arch_ids(MACHINE(pcms));

    for (i = 0; i < cpus->len; i++) {

        unsigned int apic_id = cpus->cpus[i].arch_id;

        assert(apic_id < pcms->apic_id_limit);

        if (cpus->cpus[i].props.has_node_id) {

            numa_fw_cfg[apic_id + 1] = cpu_to_le64(cpus->cpus[i].props.node_id);

        }

    }

    for (i = 0; i < nb_numa_nodes; i++) {

        numa_fw_cfg[pcms->apic_id_limit + 1 + i] =

            cpu_to_le64(numa_info[i].node_mem);

    }

    fw_cfg_add_bytes(fw_cfg, FW_CFG_NUMA, numa_fw_cfg,

                     (1 + pcms->apic_id_limit + nb_numa_nodes) *

                     sizeof(*numa_fw_cfg));



    return fw_cfg;

}
