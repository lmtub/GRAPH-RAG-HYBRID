static void esp_pci_io_write(void *opaque, target_phys_addr_t addr,

                             uint64_t val, unsigned int size)

{

    PCIESPState *pci = opaque;



    if (size < 4 || addr & 3) {

        /* need to upgrade request: we only support 4-bytes accesses */

        uint32_t current = 0, mask;

        int shift;



        if (addr < 0x40) {

            current = pci->esp.wregs[addr >> 2];

        } else if (addr < 0x60) {

            current = pci->dma_regs[(addr - 0x40) >> 2];

        } else if (addr < 0x74) {

            current = pci->sbac;

        }



        shift = (4 - size) * 8;

        mask = (~(uint32_t)0 << shift) >> shift;



        shift = ((4 - (addr & 3)) & 3) * 8;

        val <<= shift;

        val |= current & ~(mask << shift);

        addr &= ~3;

        size = 4;

    }



    if (addr < 0x40) {

        /* SCSI core reg */

        esp_reg_write(&pci->esp, addr >> 2, val);

    } else if (addr < 0x60) {

        /* PCI DMA CCB */

        esp_pci_dma_write(pci, (addr - 0x40) >> 2, val);

    } else if (addr == 0x70) {

        /* DMA SCSI Bus and control */

        trace_esp_pci_sbac_write(pci->sbac, val);

        pci->sbac = val;

    } else {

        trace_esp_pci_error_invalid_write((int)addr);

    }

}
