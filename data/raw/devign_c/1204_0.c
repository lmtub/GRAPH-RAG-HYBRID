static uint64_t ahci_mem_read(void *opaque, target_phys_addr_t addr,

                              unsigned size)

{

    AHCIState *s = opaque;

    uint32_t val = 0;



    if (addr < AHCI_GENERIC_HOST_CONTROL_REGS_MAX_ADDR) {

        switch (addr) {

        case HOST_CAP:

            val = s->control_regs.cap;

            break;

        case HOST_CTL:

            val = s->control_regs.ghc;

            break;

        case HOST_IRQ_STAT:

            val = s->control_regs.irqstatus;

            break;

        case HOST_PORTS_IMPL:

            val = s->control_regs.impl;

            break;

        case HOST_VERSION:

            val = s->control_regs.version;

            break;

        }



        DPRINTF(-1, "(addr 0x%08X), val 0x%08X\n", (unsigned) addr, val);

    } else if ((addr >= AHCI_PORT_REGS_START_ADDR) &&

               (addr < (AHCI_PORT_REGS_START_ADDR +

                (s->ports * AHCI_PORT_ADDR_OFFSET_LEN)))) {

        val = ahci_port_read(s, (addr - AHCI_PORT_REGS_START_ADDR) >> 7,

                             addr & AHCI_PORT_ADDR_OFFSET_MASK);

    }



    return val;

}
