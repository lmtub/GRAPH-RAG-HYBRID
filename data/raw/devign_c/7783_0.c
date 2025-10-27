bool memory_region_present(MemoryRegion *parent, hwaddr addr)

{

    MemoryRegion *mr = memory_region_find(parent, addr, 1).mr;

    if (!mr) {

        return false;

    }

    memory_region_unref(mr);

    return true;

}
