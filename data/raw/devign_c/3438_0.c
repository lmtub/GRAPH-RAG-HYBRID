static void gt64120_writel (void *opaque, target_phys_addr_t addr,

                            uint64_t val, unsigned size)

{

    GT64120State *s = opaque;

    PCIHostState *phb = PCI_HOST_BRIDGE(s);

    uint32_t saddr;



    if (!(s->regs[GT_CPU] & 0x00001000))

        val = bswap32(val);



    saddr = (addr & 0xfff) >> 2;

    switch (saddr) {



    /* CPU Configuration */

    case GT_CPU:

        s->regs[GT_CPU] = val;

        break;

    case GT_MULTI:

	/* Read-only register as only one GT64xxx is present on the CPU bus */

        break;



    /* CPU Address Decode */

    case GT_PCI0IOLD:

        s->regs[GT_PCI0IOLD]    = val & 0x00007fff;

        s->regs[GT_PCI0IOREMAP] = val & 0x000007ff;

        gt64120_pci_mapping(s);

        break;

    case GT_PCI0M0LD:

        s->regs[GT_PCI0M0LD]    = val & 0x00007fff;

        s->regs[GT_PCI0M0REMAP] = val & 0x000007ff;

        break;

    case GT_PCI0M1LD:

        s->regs[GT_PCI0M1LD]    = val & 0x00007fff;

        s->regs[GT_PCI0M1REMAP] = val & 0x000007ff;

        break;

    case GT_PCI1IOLD:

        s->regs[GT_PCI1IOLD]    = val & 0x00007fff;

        s->regs[GT_PCI1IOREMAP] = val & 0x000007ff;

        break;

    case GT_PCI1M0LD:

        s->regs[GT_PCI1M0LD]    = val & 0x00007fff;

        s->regs[GT_PCI1M0REMAP] = val & 0x000007ff;

        break;

    case GT_PCI1M1LD:

        s->regs[GT_PCI1M1LD]    = val & 0x00007fff;

        s->regs[GT_PCI1M1REMAP] = val & 0x000007ff;

        break;

    case GT_PCI0IOHD:

        s->regs[saddr] = val & 0x0000007f;

        gt64120_pci_mapping(s);

        break;

    case GT_PCI0M0HD:

    case GT_PCI0M1HD:

    case GT_PCI1IOHD:

    case GT_PCI1M0HD:

    case GT_PCI1M1HD:

        s->regs[saddr] = val & 0x0000007f;

        break;

    case GT_ISD:

        s->regs[saddr] = val & 0x00007fff;

        gt64120_isd_mapping(s);

        break;



    case GT_PCI0IOREMAP:

    case GT_PCI0M0REMAP:

    case GT_PCI0M1REMAP:

    case GT_PCI1IOREMAP:

    case GT_PCI1M0REMAP:

    case GT_PCI1M1REMAP:

        s->regs[saddr] = val & 0x000007ff;

        break;



    /* CPU Error Report */

    case GT_CPUERR_ADDRLO:

    case GT_CPUERR_ADDRHI:

    case GT_CPUERR_DATALO:

    case GT_CPUERR_DATAHI:

    case GT_CPUERR_PARITY:

	/* Read-only registers, do nothing */

        break;



    /* CPU Sync Barrier */

    case GT_PCI0SYNC:

    case GT_PCI1SYNC:

	/* Read-only registers, do nothing */

        break;



    /* SDRAM and Device Address Decode */

    case GT_SCS0LD:

    case GT_SCS0HD:

    case GT_SCS1LD:

    case GT_SCS1HD:

    case GT_SCS2LD:

    case GT_SCS2HD:

    case GT_SCS3LD:

    case GT_SCS3HD:

    case GT_CS0LD:

    case GT_CS0HD:

    case GT_CS1LD:

    case GT_CS1HD:

    case GT_CS2LD:

    case GT_CS2HD:

    case GT_CS3LD:

    case GT_CS3HD:

    case GT_BOOTLD:

    case GT_BOOTHD:

    case GT_ADERR:

    /* SDRAM Configuration */

    case GT_SDRAM_CFG:

    case GT_SDRAM_OPMODE:

    case GT_SDRAM_BM:

    case GT_SDRAM_ADDRDECODE:

        /* Accept and ignore SDRAM interleave configuration */

        s->regs[saddr] = val;

        break;



    /* Device Parameters */

    case GT_DEV_B0:

    case GT_DEV_B1:

    case GT_DEV_B2:

    case GT_DEV_B3:

    case GT_DEV_BOOT:

        /* Not implemented */

        DPRINTF ("Unimplemented device register offset 0x%x\n", saddr << 2);

        break;



    /* ECC */

    case GT_ECC_ERRDATALO:

    case GT_ECC_ERRDATAHI:

    case GT_ECC_MEM:

    case GT_ECC_CALC:

    case GT_ECC_ERRADDR:

        /* Read-only registers, do nothing */

        break;



    /* DMA Record */

    case GT_DMA0_CNT:

    case GT_DMA1_CNT:

    case GT_DMA2_CNT:

    case GT_DMA3_CNT:

    case GT_DMA0_SA:

    case GT_DMA1_SA:

    case GT_DMA2_SA:

    case GT_DMA3_SA:

    case GT_DMA0_DA:

    case GT_DMA1_DA:

    case GT_DMA2_DA:

    case GT_DMA3_DA:

    case GT_DMA0_NEXT:

    case GT_DMA1_NEXT:

    case GT_DMA2_NEXT:

    case GT_DMA3_NEXT:

    case GT_DMA0_CUR:

    case GT_DMA1_CUR:

    case GT_DMA2_CUR:

    case GT_DMA3_CUR:

        /* Not implemented */

        DPRINTF ("Unimplemented DMA register offset 0x%x\n", saddr << 2);

        break;



    /* DMA Channel Control */

    case GT_DMA0_CTRL:

    case GT_DMA1_CTRL:

    case GT_DMA2_CTRL:

    case GT_DMA3_CTRL:

        /* Not implemented */

        DPRINTF ("Unimplemented DMA register offset 0x%x\n", saddr << 2);

        break;



    /* DMA Arbiter */

    case GT_DMA_ARB:

        /* Not implemented */

        DPRINTF ("Unimplemented DMA register offset 0x%x\n", saddr << 2);

        break;



    /* Timer/Counter */

    case GT_TC0:

    case GT_TC1:

    case GT_TC2:

    case GT_TC3:

    case GT_TC_CONTROL:

        /* Not implemented */

        DPRINTF ("Unimplemented timer register offset 0x%x\n", saddr << 2);

        break;



    /* PCI Internal */

    case GT_PCI0_CMD:

    case GT_PCI1_CMD:

        s->regs[saddr] = val & 0x0401fc0f;

        break;

    case GT_PCI0_TOR:

    case GT_PCI0_BS_SCS10:

    case GT_PCI0_BS_SCS32:

    case GT_PCI0_BS_CS20:

    case GT_PCI0_BS_CS3BT:

    case GT_PCI1_IACK:

    case GT_PCI0_IACK:

    case GT_PCI0_BARE:

    case GT_PCI0_PREFMBR:

    case GT_PCI0_SCS10_BAR:

    case GT_PCI0_SCS32_BAR:

    case GT_PCI0_CS20_BAR:

    case GT_PCI0_CS3BT_BAR:

    case GT_PCI0_SSCS10_BAR:

    case GT_PCI0_SSCS32_BAR:

    case GT_PCI0_SCS3BT_BAR:

    case GT_PCI1_TOR:

    case GT_PCI1_BS_SCS10:

    case GT_PCI1_BS_SCS32:

    case GT_PCI1_BS_CS20:

    case GT_PCI1_BS_CS3BT:

    case GT_PCI1_BARE:

    case GT_PCI1_PREFMBR:

    case GT_PCI1_SCS10_BAR:

    case GT_PCI1_SCS32_BAR:

    case GT_PCI1_CS20_BAR:

    case GT_PCI1_CS3BT_BAR:

    case GT_PCI1_SSCS10_BAR:

    case GT_PCI1_SSCS32_BAR:

    case GT_PCI1_SCS3BT_BAR:

    case GT_PCI1_CFGADDR:

    case GT_PCI1_CFGDATA:

        /* not implemented */

        break;

    case GT_PCI0_CFGADDR:

        phb->config_reg = val & 0x80fffffc;

        break;

    case GT_PCI0_CFGDATA:

        if (!(s->regs[GT_PCI0_CMD] & 1) && (phb->config_reg & 0x00fff800)) {

            val = bswap32(val);

        }

        if (phb->config_reg & (1u << 31)) {

            pci_data_write(phb->bus, phb->config_reg, val, 4);

        }

        break;



    /* Interrupts */

    case GT_INTRCAUSE:

        /* not really implemented */

        s->regs[saddr] = ~(~(s->regs[saddr]) | ~(val & 0xfffffffe));

        s->regs[saddr] |= !!(s->regs[saddr] & 0xfffffffe);

        DPRINTF("INTRCAUSE %" PRIx64 "\n", val);

        break;

    case GT_INTRMASK:

        s->regs[saddr] = val & 0x3c3ffffe;

        DPRINTF("INTRMASK %" PRIx64 "\n", val);

        break;

    case GT_PCI0_ICMASK:

        s->regs[saddr] = val & 0x03fffffe;

        DPRINTF("ICMASK %" PRIx64 "\n", val);

        break;

    case GT_PCI0_SERR0MASK:

        s->regs[saddr] = val & 0x0000003f;

        DPRINTF("SERR0MASK %" PRIx64 "\n", val);

        break;



    /* Reserved when only PCI_0 is configured. */

    case GT_HINTRCAUSE:

    case GT_CPU_INTSEL:

    case GT_PCI0_INTSEL:

    case GT_HINTRMASK:

    case GT_PCI0_HICMASK:

    case GT_PCI1_SERR1MASK:

        /* not implemented */

        break;



    /* SDRAM Parameters */

    case GT_SDRAM_B0:

    case GT_SDRAM_B1:

    case GT_SDRAM_B2:

    case GT_SDRAM_B3:

        /* We don't simulate electrical parameters of the SDRAM.

           Accept, but ignore the values. */

        s->regs[saddr] = val;

        break;



    default:

        DPRINTF ("Bad register offset 0x%x\n", (int)addr);

        break;

    }

}
