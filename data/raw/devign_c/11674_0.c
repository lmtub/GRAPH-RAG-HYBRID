static inline bool memory_access_is_direct(MemoryRegion *mr, bool is_write)

{

    if (memory_region_is_ram(mr)) {

        return !(is_write && mr->readonly);

    }

    if (memory_region_is_romd(mr)) {

        return !is_write;

    }



    return false;

}
