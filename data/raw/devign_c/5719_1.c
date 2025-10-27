static MemTxResult  memory_region_read_accessor(MemoryRegion *mr,

                                                hwaddr addr,

                                                uint64_t *value,

                                                unsigned size,

                                                unsigned shift,

                                                uint64_t mask,

                                                MemTxAttrs attrs)

{

    uint64_t tmp;



    tmp = mr->ops->read(mr->opaque, addr, size);

    if (mr->subpage) {

        trace_memory_region_subpage_read(get_cpu_index(), mr, addr, tmp, size);






    } else if (TRACE_MEMORY_REGION_OPS_READ_ENABLED) {

        hwaddr abs_addr = memory_region_to_absolute_addr(mr, addr);

        trace_memory_region_ops_read(get_cpu_index(), mr, abs_addr, tmp, size);

    }

    *value |= (tmp & mask) << shift;

    return MEMTX_OK;

}