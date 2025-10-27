static void pci_reset(EEPRO100State * s)

{

    uint32_t device = s->device;

    uint8_t *pci_conf = s->pci_dev->config;



    logout("%p\n", s);



    /* PCI Vendor ID */

    pci_config_set_vendor_id(pci_conf, PCI_VENDOR_ID_INTEL);

    /* PCI Device ID */

    pci_config_set_device_id(pci_conf, PCI_DEVICE_ID_INTEL_82551IT);

    /* PCI Command */

    PCI_CONFIG_16(PCI_COMMAND, 0x0000);

    /* PCI Status */

    PCI_CONFIG_16(PCI_STATUS, 0x2800);

    /* PCI Revision ID */

    PCI_CONFIG_8(PCI_REVISION_ID, 0x08);

    /* PCI Class Code */

    PCI_CONFIG_8(0x09, 0x00);

    pci_config_set_class(pci_conf, PCI_CLASS_NETWORK_ETHERNET);

    /* PCI Cache Line Size */

    /* check cache line size!!! */

    //~ PCI_CONFIG_8(0x0c, 0x00);

    /* PCI Latency Timer */

    PCI_CONFIG_8(0x0d, 0x20);   // latency timer = 32 clocks

    /* PCI Header Type */

    /* BIST (built-in self test) */

#if defined(TARGET_I386)

// !!! workaround for buggy bios

//~ #define PCI_ADDRESS_SPACE_MEM_PREFETCH 0

#endif

#if 0

    /* PCI Base Address Registers */

    /* CSR Memory Mapped Base Address */

    PCI_CONFIG_32(PCI_BASE_ADDRESS_0,

                  PCI_ADDRESS_SPACE_MEM | PCI_ADDRESS_SPACE_MEM_PREFETCH);

    /* CSR I/O Mapped Base Address */

    PCI_CONFIG_32(PCI_BASE_ADDRESS_1, PCI_ADDRESS_SPACE_IO);

#if 0

    /* Flash Memory Mapped Base Address */

    PCI_CONFIG_32(PCI_BASE_ADDRESS_2, 0xfffe0000 | PCI_ADDRESS_SPACE_MEM);

#endif

#endif

    /* Expansion ROM Base Address (depends on boot disable!!!) */

    PCI_CONFIG_32(0x30, 0x00000000);

    /* Capability Pointer */

    PCI_CONFIG_8(0x34, 0xdc);

    /* Interrupt Pin */

    PCI_CONFIG_8(0x3d, 1);      // interrupt pin 0

    /* Minimum Grant */

    PCI_CONFIG_8(0x3e, 0x08);

    /* Maximum Latency */

    PCI_CONFIG_8(0x3f, 0x18);

    /* Power Management Capabilities / Next Item Pointer / Capability ID */

    PCI_CONFIG_32(0xdc, 0x7e210001);



    switch (device) {

    case i82551:

        //~ PCI_CONFIG_16(PCI_DEVICE_ID, 0x1209);

        PCI_CONFIG_8(PCI_REVISION_ID, 0x0f);

        break;

    case i82557B:

        PCI_CONFIG_16(PCI_DEVICE_ID, 0x1229);

        PCI_CONFIG_8(PCI_REVISION_ID, 0x02);

        break;

    case i82557C:

        PCI_CONFIG_16(PCI_DEVICE_ID, 0x1229);

        PCI_CONFIG_8(PCI_REVISION_ID, 0x03);

        break;

    case i82558B:

        PCI_CONFIG_16(PCI_DEVICE_ID, 0x1229);

        PCI_CONFIG_16(PCI_STATUS, 0x2810);

        PCI_CONFIG_8(PCI_REVISION_ID, 0x05);

        break;

    case i82559C:

        PCI_CONFIG_16(PCI_DEVICE_ID, 0x1229);

        PCI_CONFIG_16(PCI_STATUS, 0x2810);

        //~ PCI_CONFIG_8(PCI_REVISION_ID, 0x08);

        break;

    case i82559ER:

        //~ PCI_CONFIG_16(PCI_DEVICE_ID, 0x1209);

        PCI_CONFIG_16(PCI_STATUS, 0x2810);

        PCI_CONFIG_8(PCI_REVISION_ID, 0x09);

        break;

    //~ PCI_CONFIG_16(PCI_DEVICE_ID, 0x1029);

    //~ PCI_CONFIG_16(PCI_DEVICE_ID, 0x1030);       /* 82559 InBusiness 10/100 */

    default:

        logout("Device %X is undefined!\n", device);

    }



    if (device == i82557C || device == i82558B || device == i82559C) {

        logout("Get device id and revision from EEPROM!!!\n");

    }

}
