static void memory_region_write_thunk_n(void *_mr,

                                        target_phys_addr_t addr,

                                        unsigned size,

                                        uint64_t data)

{

    MemoryRegion *mr = _mr;



    if (!memory_region_access_valid(mr, addr, size)) {

        return; /* FIXME: better signalling */

    }



    if (!mr->ops->write) {

        mr->ops->old_mmio.write[bitops_ffsl(size)](mr->opaque, addr, data);

        return;

    }



    /* FIXME: support unaligned access */

    access_with_adjusted_size(addr + mr->offset, &data, size,

                              mr->ops->impl.min_access_size,

                              mr->ops->impl.max_access_size,

                              memory_region_write_accessor, mr);

}
