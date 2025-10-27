static void sysbus_mmio_map_common(SysBusDevice *dev, int n, hwaddr addr,

                                   bool may_overlap, unsigned priority)

{

    assert(n >= 0 && n < dev->num_mmio);



    if (dev->mmio[n].addr == addr) {

        /* ??? region already mapped here.  */

        return;

    }

    if (dev->mmio[n].addr != (hwaddr)-1) {

        /* Unregister previous mapping.  */

        memory_region_del_subregion(get_system_memory(), dev->mmio[n].memory);

    }

    dev->mmio[n].addr = addr;

    if (may_overlap) {

        memory_region_add_subregion_overlap(get_system_memory(),

                                            addr,

                                            dev->mmio[n].memory,

                                            priority);

    }

    else {

        memory_region_add_subregion(get_system_memory(),

                                    addr,

                                    dev->mmio[n].memory);

    }

}
