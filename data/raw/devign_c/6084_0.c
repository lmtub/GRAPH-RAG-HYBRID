static void memory_region_iorange_write(IORange *iorange,

                                        uint64_t offset,

                                        unsigned width,

                                        uint64_t data)

{

    MemoryRegionIORange *mrio

        = container_of(iorange, MemoryRegionIORange, iorange);

    MemoryRegion *mr = mrio->mr;



    offset += mrio->offset;

    if (mr->ops->old_portio) {

        const MemoryRegionPortio *mrp = find_portio(mr, offset - mrio->offset,

                                                    width, true);



        if (mrp) {

            mrp->write(mr->opaque, offset, data);

        } else if (width == 2) {

            mrp = find_portio(mr, offset - mrio->offset, 1, true);

            assert(mrp);

            mrp->write(mr->opaque, offset, data & 0xff);

            mrp->write(mr->opaque, offset + 1, data >> 8);

        }

        return;

    }

    access_with_adjusted_size(offset, &data, width,

                              mr->ops->impl.min_access_size,

                              mr->ops->impl.max_access_size,

                              memory_region_write_accessor, mr);

}
