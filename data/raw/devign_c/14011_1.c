static void memory_region_oldmmio_write_accessor(MemoryRegion *mr,

                                                 hwaddr addr,

                                                 uint64_t *value,

                                                 unsigned size,

                                                 unsigned shift,

                                                 uint64_t mask)

{

    uint64_t tmp;



    tmp = (*value >> shift) & mask;

    trace_memory_region_ops_write(mr, addr, tmp, size);

    mr->ops->old_mmio.write[ctz32(size)](mr->opaque, addr, tmp);

}
