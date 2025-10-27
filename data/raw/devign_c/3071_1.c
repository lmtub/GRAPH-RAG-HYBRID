bool io_mem_read(MemoryRegion *mr, hwaddr addr, uint64_t *pval, unsigned size)

{

    return memory_region_dispatch_read(mr, addr, pval, size);

}
