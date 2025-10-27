static uint16_t pci_req_id_cache_extract(PCIReqIDCache *cache)

{

    uint8_t bus_n;

    uint16_t result;



    switch (cache->type) {

    case PCI_REQ_ID_BDF:

        result = pci_get_bdf(cache->dev);

        break;

    case PCI_REQ_ID_SECONDARY_BUS:

        bus_n = pci_bus_num(cache->dev->bus);

        result = PCI_BUILD_BDF(bus_n, 0);

        break;

    default:

        error_printf("Invalid PCI requester ID cache type: %d\n",

                     cache->type);

        exit(1);

        break;

    }



    return result;

}
