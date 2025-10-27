void memory_region_add_subregion_overlap(MemoryRegion *mr,

                                         hwaddr offset,

                                         MemoryRegion *subregion,

                                         int priority)

{

    subregion->may_overlap = true;

    subregion->priority = priority;

    memory_region_add_subregion_common(mr, offset, subregion);

}
