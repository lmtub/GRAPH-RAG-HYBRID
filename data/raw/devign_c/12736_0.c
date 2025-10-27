build_spcr(GArray *table_data, GArray *linker, VirtGuestInfo *guest_info)

{

    AcpiSerialPortConsoleRedirection *spcr;

    const MemMapEntry *uart_memmap = &guest_info->memmap[VIRT_UART];

    int irq = guest_info->irqmap[VIRT_UART] + ARM_SPI_BASE;



    spcr = acpi_data_push(table_data, sizeof(*spcr));



    spcr->interface_type = 0x3;    /* ARM PL011 UART */



    spcr->base_address.space_id = AML_SYSTEM_MEMORY;

    spcr->base_address.bit_width = 8;

    spcr->base_address.bit_offset = 0;

    spcr->base_address.access_width = 1;

    spcr->base_address.address = cpu_to_le64(uart_memmap->base);



    spcr->interrupt_types = (1 << 3); /* Bit[3] ARMH GIC interrupt */

    spcr->gsi = cpu_to_le32(irq);  /* Global System Interrupt */



    spcr->baud = 3;                /* Baud Rate: 3 = 9600 */

    spcr->parity = 0;              /* No Parity */

    spcr->stopbits = 1;            /* 1 Stop bit */

    spcr->flowctrl = (1 << 1);     /* Bit[1] = RTS/CTS hardware flow control */

    spcr->term_type = 0;           /* Terminal Type: 0 = VT100 */



    spcr->pci_device_id = 0xffff;  /* PCI Device ID: not a PCI device */

    spcr->pci_vendor_id = 0xffff;  /* PCI Vendor ID: not a PCI device */



    build_header(linker, table_data, (void *)spcr, "SPCR", sizeof(*spcr), 2,

                 NULL);

}
