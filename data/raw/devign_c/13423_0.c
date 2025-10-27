static AddressSpace *memory_region_to_address_space(MemoryRegion *mr)

{

    AddressSpace *as;



    while (mr->container) {

        mr = mr->container;

    }

    QTAILQ_FOREACH(as, &address_spaces, address_spaces_link) {

        if (mr == as->root) {

            return as;

        }

    }

    abort();

}
