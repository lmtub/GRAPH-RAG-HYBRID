bool memory_region_is_logging(MemoryRegion *mr)

{

    return mr->dirty_log_mask;

}
