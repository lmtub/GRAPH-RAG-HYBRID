void sysbus_mmio_map_overlap(SysBusDevice *dev, int n, hwaddr addr,

                             unsigned priority)

{

    sysbus_mmio_map_common(dev, n, addr, true, priority);

}
