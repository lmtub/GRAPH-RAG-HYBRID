static inline int memory_access_size(MemoryRegion *mr, int l, hwaddr addr)

{

    if (l >= 4 && (((addr & 3) == 0 || mr->ops->impl.unaligned))) {

        return 4;

    }

    if (l >= 2 && (((addr & 1) == 0) || mr->ops->impl.unaligned)) {

        return 2;

    }

    return 1;

}
