void spapr_create_phb(sPAPREnvironment *spapr,

                      const char *busname, uint64_t buid,

                      uint64_t mem_win_addr, uint64_t mem_win_size,

                      uint64_t io_win_addr, uint64_t msi_win_addr)

{

    DeviceState *dev;



    dev = qdev_create(NULL, TYPE_SPAPR_PCI_HOST_BRIDGE);



    if (busname) {

        qdev_prop_set_string(dev, "busname", g_strdup(busname));

    }

    qdev_prop_set_uint64(dev, "buid", buid);

    qdev_prop_set_uint64(dev, "mem_win_addr", mem_win_addr);

    qdev_prop_set_uint64(dev, "mem_win_size", mem_win_size);

    qdev_prop_set_uint64(dev, "io_win_addr", io_win_addr);

    qdev_prop_set_uint64(dev, "msi_win_addr", msi_win_addr);



    qdev_init_nofail(dev);

}
