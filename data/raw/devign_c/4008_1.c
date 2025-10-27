MemTxResult memory_region_dispatch_write(MemoryRegion *mr,
                                         hwaddr addr,
                                         uint64_t data,
                                         unsigned size,
                                         MemTxAttrs attrs)
{
    if (!memory_region_access_valid(mr, addr, size, true)) {
        unassigned_mem_write(mr, addr, data, size);
        return MEMTX_DECODE_ERROR;
    adjust_endianness(mr, &data, size);
    if (mr->ops->write) {
        return access_with_adjusted_size(addr, &data, size,
                                         mr->ops->impl.min_access_size,
                                         mr->ops->impl.max_access_size,
                                         memory_region_write_accessor, mr,
                                         attrs);
    } else if (mr->ops->write_with_attrs) {
        return
            access_with_adjusted_size(addr, &data, size,
                                      mr->ops->impl.min_access_size,
                                      mr->ops->impl.max_access_size,
                                      memory_region_write_with_attrs_accessor,
                                      mr, attrs);
    } else {
        return access_with_adjusted_size(addr, &data, size, 1, 4,
                                         memory_region_oldmmio_write_accessor,
                                         mr, attrs);