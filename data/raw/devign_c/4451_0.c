static const MemoryRegionPortio *find_portio(MemoryRegion *mr, uint64_t offset,

                                             unsigned width, bool write)

{

    const MemoryRegionPortio *mrp;



    for (mrp = mr->ops->old_portio; mrp->size; ++mrp) {

        if (offset >= mrp->offset && offset < mrp->offset + mrp->len

            && width == mrp->size

            && (write ? (bool)mrp->write : (bool)mrp->read)) {

            return mrp;

        }

    }

    return NULL;

}
