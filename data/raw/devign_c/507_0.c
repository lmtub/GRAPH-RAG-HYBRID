static void *vring_map(MemoryRegion **mr, hwaddr phys, hwaddr len,

                       bool is_write)

{

    MemoryRegionSection section = memory_region_find(get_system_memory(), phys, len);



    if (!section.mr || int128_get64(section.size) < len) {

        goto out;

    }

    if (is_write && section.readonly) {

        goto out;

    }

    if (!memory_region_is_ram(section.mr)) {

        goto out;

    }



    /* Ignore regions with dirty logging, we cannot mark them dirty */

    if (memory_region_is_logging(section.mr)) {

        goto out;

    }



    *mr = section.mr;

    return memory_region_get_ram_ptr(section.mr) + section.offset_within_region;



out:

    memory_region_unref(section.mr);

    *mr = NULL;

    return NULL;

}
