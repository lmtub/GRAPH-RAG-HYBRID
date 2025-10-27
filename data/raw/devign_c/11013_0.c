build_srat(GArray *table_data, GArray *linker, VirtGuestInfo *guest_info)

{

    AcpiSystemResourceAffinityTable *srat;

    AcpiSratProcessorGiccAffinity *core;

    AcpiSratMemoryAffinity *numamem;

    int i, j, srat_start;

    uint64_t mem_base;

    uint32_t *cpu_node = g_malloc0(guest_info->smp_cpus * sizeof(uint32_t));



    for (i = 0; i < guest_info->smp_cpus; i++) {

        for (j = 0; j < nb_numa_nodes; j++) {

            if (test_bit(i, numa_info[j].node_cpu)) {

                cpu_node[i] = j;

                break;

            }

        }

    }



    srat_start = table_data->len;

    srat = acpi_data_push(table_data, sizeof(*srat));

    srat->reserved1 = cpu_to_le32(1);



    for (i = 0; i < guest_info->smp_cpus; ++i) {

        core = acpi_data_push(table_data, sizeof(*core));

        core->type = ACPI_SRAT_PROCESSOR_GICC;

        core->length = sizeof(*core);

        core->proximity = cpu_to_le32(cpu_node[i]);

        core->acpi_processor_uid = cpu_to_le32(i);

        core->flags = cpu_to_le32(1);

    }

    g_free(cpu_node);



    mem_base = guest_info->memmap[VIRT_MEM].base;

    for (i = 0; i < nb_numa_nodes; ++i) {

        numamem = acpi_data_push(table_data, sizeof(*numamem));

        build_srat_memory(numamem, mem_base, numa_info[i].node_mem, i,

                          MEM_AFFINITY_ENABLED);

        mem_base += numa_info[i].node_mem;

    }



    build_header(linker, table_data,

                 (void *)(table_data->data + srat_start), "SRAT",

                 table_data->len - srat_start, 3, NULL, NULL);

}
