static void read_storage_element0_info(SCLPDevice *sclp, SCCB *sccb)

{

    int i, assigned;

    int subincrement_id = SCLP_STARTING_SUBINCREMENT_ID;

    ReadStorageElementInfo *storage_info = (ReadStorageElementInfo *) sccb;

    sclpMemoryHotplugDev *mhd = get_sclp_memory_hotplug_dev();



    assert(mhd);



    if ((ram_size >> mhd->increment_size) >= 0x10000) {

        sccb->h.response_code = cpu_to_be16(SCLP_RC_SCCB_BOUNDARY_VIOLATION);

        return;

    }



    /* Return information regarding core memory */

    storage_info->max_id = cpu_to_be16(mhd->standby_mem_size ? 1 : 0);

    assigned = ram_size >> mhd->increment_size;

    storage_info->assigned = cpu_to_be16(assigned);



    for (i = 0; i < assigned; i++) {

        storage_info->entries[i] = cpu_to_be32(subincrement_id);

        subincrement_id += SCLP_INCREMENT_UNIT;

    }

    sccb->h.response_code = cpu_to_be16(SCLP_RC_NORMAL_READ_COMPLETION);

}
