build_srat(GArray *table_data, GArray *linker)

{

    AcpiSystemResourceAffinityTable *srat;

    AcpiSratProcessorAffinity *core;

    AcpiSratMemoryAffinity *numamem;



    int i;

    uint64_t curnode;

    int srat_start, numa_start, slots;

    uint64_t mem_len, mem_base, next_base;

    PCMachineState *pcms = PC_MACHINE(qdev_get_machine());

    ram_addr_t hotplugabble_address_space_size =

        object_property_get_int(OBJECT(pcms), PC_MACHINE_MEMHP_REGION_SIZE,

                                NULL);



    srat_start = table_data->len;



    srat = acpi_data_push(table_data, sizeof *srat);

    srat->reserved1 = cpu_to_le32(1);

    core = (void *)(srat + 1);



    for (i = 0; i < pcms->apic_id_limit; ++i) {

        core = acpi_data_push(table_data, sizeof *core);

        core->type = ACPI_SRAT_PROCESSOR;

        core->length = sizeof(*core);

        core->local_apic_id = i;

        curnode = pcms->node_cpu[i];

        core->proximity_lo = curnode;

        memset(core->proximity_hi, 0, 3);

        core->local_sapic_eid = 0;

        core->flags = cpu_to_le32(1);

    }





    /* the memory map is a bit tricky, it contains at least one hole

     * from 640k-1M and possibly another one from 3.5G-4G.

     */

    next_base = 0;

    numa_start = table_data->len;



    numamem = acpi_data_push(table_data, sizeof *numamem);

    acpi_build_srat_memory(numamem, 0, 640*1024, 0, MEM_AFFINITY_ENABLED);

    next_base = 1024 * 1024;

    for (i = 1; i < pcms->numa_nodes + 1; ++i) {

        mem_base = next_base;

        mem_len = pcms->node_mem[i - 1];

        if (i == 1) {

            mem_len -= 1024 * 1024;

        }

        next_base = mem_base + mem_len;



        /* Cut out the ACPI_PCI hole */

        if (mem_base <= pcms->below_4g_mem_size &&

            next_base > pcms->below_4g_mem_size) {

            mem_len -= next_base - pcms->below_4g_mem_size;

            if (mem_len > 0) {

                numamem = acpi_data_push(table_data, sizeof *numamem);

                acpi_build_srat_memory(numamem, mem_base, mem_len, i - 1,

                                       MEM_AFFINITY_ENABLED);

            }

            mem_base = 1ULL << 32;

            mem_len = next_base - pcms->below_4g_mem_size;

            next_base += (1ULL << 32) - pcms->below_4g_mem_size;

        }

        numamem = acpi_data_push(table_data, sizeof *numamem);

        acpi_build_srat_memory(numamem, mem_base, mem_len, i - 1,

                               MEM_AFFINITY_ENABLED);

    }

    slots = (table_data->len - numa_start) / sizeof *numamem;

    for (; slots < pcms->numa_nodes + 2; slots++) {

        numamem = acpi_data_push(table_data, sizeof *numamem);

        acpi_build_srat_memory(numamem, 0, 0, 0, MEM_AFFINITY_NOFLAGS);

    }



    /*

     * Entry is required for Windows to enable memory hotplug in OS.

     * Memory devices may override proximity set by this entry,

     * providing _PXM method if necessary.

     */

    if (hotplugabble_address_space_size) {

        numamem = acpi_data_push(table_data, sizeof *numamem);

        acpi_build_srat_memory(numamem, pcms->hotplug_memory.base,

                               hotplugabble_address_space_size, 0,

                               MEM_AFFINITY_HOTPLUGGABLE |

                               MEM_AFFINITY_ENABLED);

    }



    build_header(linker, table_data,

                 (void *)(table_data->data + srat_start),

                 "SRAT",

                 table_data->len - srat_start, 1, NULL);

}
