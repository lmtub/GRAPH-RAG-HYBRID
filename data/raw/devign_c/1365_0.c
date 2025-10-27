void memory_region_set_address(MemoryRegion *mr, hwaddr addr)

{

    MemoryRegion *parent = mr->parent;

    int priority = mr->priority;

    bool may_overlap = mr->may_overlap;



    if (addr == mr->addr || !parent) {

        mr->addr = addr;

        return;

    }



    memory_region_transaction_begin();

    memory_region_ref(mr);

    memory_region_del_subregion(parent, mr);

    if (may_overlap) {

        memory_region_add_subregion_overlap(parent, addr, mr, priority);

    } else {

        memory_region_add_subregion(parent, addr, mr);

    }

    memory_region_unref(mr);

    memory_region_transaction_commit();

}
