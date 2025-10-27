static void memory_region_iorange_read(IORange *iorange,

                                       uint64_t offset,

                                       unsigned width,

                                       uint64_t *data)

{

    MemoryRegion *mr = container_of(iorange, MemoryRegion, iorange);



    if (mr->ops->old_portio) {

        const MemoryRegionPortio *mrp = find_portio(mr, offset, width, false);



        *data = ((uint64_t)1 << (width * 8)) - 1;

        if (mrp) {

            *data = mrp->read(mr->opaque, offset - mrp->offset);

        }

        return;

    }

    *data = mr->ops->read(mr->opaque, offset, width);

}
