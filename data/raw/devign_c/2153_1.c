static void assign_storage(SCLPDevice *sclp, SCCB *sccb)

{

    MemoryRegion *mr = NULL;

    uint64_t this_subregion_size;

    AssignStorage *assign_info = (AssignStorage *) sccb;

    sclpMemoryHotplugDev *mhd = get_sclp_memory_hotplug_dev();

    ram_addr_t assign_addr;

    MemoryRegion *sysmem = get_system_memory();



    if (!mhd) {

        sccb->h.response_code = cpu_to_be16(SCLP_RC_INVALID_SCLP_COMMAND);

        return;

    }

    assign_addr = (assign_info->rn - 1) * mhd->rzm;



    if ((assign_addr % MEM_SECTION_SIZE == 0) &&

        (assign_addr >= mhd->padded_ram_size)) {

        /* Re-use existing memory region if found */

        mr = memory_region_find(sysmem, assign_addr, 1).mr;

        memory_region_unref(mr);

        if (!mr) {



            MemoryRegion *standby_ram = g_new(MemoryRegion, 1);



            /* offset to align to standby_subregion_size for allocation */

            ram_addr_t offset = assign_addr -

                                (assign_addr - mhd->padded_ram_size)

                                % mhd->standby_subregion_size;



            /* strlen("standby.ram") + 4 (Max of KVM_MEMORY_SLOTS) +  NULL */

            char id[16];

            snprintf(id, 16, "standby.ram%d",

                     (int)((offset - mhd->padded_ram_size) /

                     mhd->standby_subregion_size) + 1);



            /* Allocate a subregion of the calculated standby_subregion_size */

            if (offset + mhd->standby_subregion_size >

                mhd->padded_ram_size + mhd->standby_mem_size) {

                this_subregion_size = mhd->padded_ram_size +

                  mhd->standby_mem_size - offset;

            } else {

                this_subregion_size = mhd->standby_subregion_size;

            }



            memory_region_init_ram(standby_ram, NULL, id, this_subregion_size, &error_abort);

            /* This is a hack to make memory hotunplug work again. Once we have

             * subdevices, we have to unparent them when unassigning memory,

             * instead of doing it via the ref count of the MemoryRegion. */

            object_ref(OBJECT(standby_ram));

            object_unparent(OBJECT(standby_ram));

            vmstate_register_ram_global(standby_ram);

            memory_region_add_subregion(sysmem, offset, standby_ram);

        }

        /* The specified subregion is no longer in standby */

        mhd->standby_state_map[(assign_addr - mhd->padded_ram_size)

                               / MEM_SECTION_SIZE] = 1;

    }

    sccb->h.response_code = cpu_to_be16(SCLP_RC_NORMAL_COMPLETION);

}
