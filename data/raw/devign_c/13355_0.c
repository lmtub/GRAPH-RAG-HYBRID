static void read_storage_element1_info(SCLPDevice *sclp, SCCB *sccb)

{

    ReadStorageElementInfo *storage_info = (ReadStorageElementInfo *) sccb;

    sclpMemoryHotplugDev *mhd = get_sclp_memory_hotplug_dev();



    assert(mhd);



    if ((mhd->standby_mem_size >> mhd->increment_size) >= 0x10000) {

        sccb->h.response_code = cpu_to_be16(SCLP_RC_SCCB_BOUNDARY_VIOLATION);

        return;

    }



    /* Return information regarding standby memory */

    storage_info->max_id = cpu_to_be16(mhd->standby_mem_size ? 1 : 0);

    storage_info->assigned = cpu_to_be16(mhd->standby_mem_size >>

                                         mhd->increment_size);

    storage_info->standby = cpu_to_be16(mhd->standby_mem_size >>

                                        mhd->increment_size);

    sccb->h.response_code = cpu_to_be16(SCLP_RC_STANDBY_READ_COMPLETION);

}
