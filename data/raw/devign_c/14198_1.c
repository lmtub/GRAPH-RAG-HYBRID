static void unassign_storage(SCCB *sccb)

{

    MemoryRegion *mr = NULL;

    AssignStorage *assign_info = (AssignStorage *) sccb;

    sclpMemoryHotplugDev *mhd = get_sclp_memory_hotplug_dev();

    assert(mhd);

    ram_addr_t unassign_addr = (assign_info->rn - 1) * mhd->rzm;

    MemoryRegion *sysmem = get_system_memory();



    /* if the addr is a multiple of 256 MB */

    if ((unassign_addr % MEM_SECTION_SIZE == 0) &&

        (unassign_addr >= mhd->padded_ram_size)) {

        mhd->standby_state_map[(unassign_addr -

                           mhd->padded_ram_size) / MEM_SECTION_SIZE] = 0;



        /* find the specified memory region and destroy it */

        mr = memory_region_find(sysmem, unassign_addr, 1).mr;

        if (mr) {

            int i;

            int is_removable = 1;

            ram_addr_t map_offset = (unassign_addr - mhd->padded_ram_size -

                                     (unassign_addr - mhd->padded_ram_size)

                                     % mhd->standby_subregion_size);

            /* Mark all affected subregions as 'standby' once again */

            for (i = 0;

                 i < (mhd->standby_subregion_size / MEM_SECTION_SIZE);

                 i++) {



                if (mhd->standby_state_map[i + map_offset / MEM_SECTION_SIZE]) {

                    is_removable = 0;

                    break;

                }

            }

            if (is_removable) {

                memory_region_del_subregion(sysmem, mr);

                object_unparent(OBJECT(mr));

                g_free(mr);

            }

        }

    }

    sccb->h.response_code = cpu_to_be16(SCLP_RC_NORMAL_COMPLETION);

}
