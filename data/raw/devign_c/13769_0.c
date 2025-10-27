void pci_default_write_config(PCIDevice *d,

                              uint32_t address, uint32_t val, int len)

{

    int can_write, i;

    uint32_t end, addr;



    if (len == 4 && ((address >= 0x10 && address < 0x10 + 4 * 6) ||

                     (address >= 0x30 && address < 0x34))) {

        PCIIORegion *r;

        int reg;



        if ( address >= 0x30 ) {

            reg = PCI_ROM_SLOT;

        }else{

            reg = (address - 0x10) >> 2;

        }

        r = &d->io_regions[reg];

        if (r->size == 0)

            goto default_config;

        /* compute the stored value */

        if (reg == PCI_ROM_SLOT) {

            /* keep ROM enable bit */

            val &= (~(r->size - 1)) | 1;

        } else {

            val &= ~(r->size - 1);

            val |= r->type;

        }

        *(uint32_t *)(d->config + address) = cpu_to_le32(val);

        pci_update_mappings(d);

        return;

    }

 default_config:

    /* not efficient, but simple */

    addr = address;

    for(i = 0; i < len; i++) {

        /* default read/write accesses */

        switch(d->config[0x0e]) {

        case 0x00:

        case 0x80:

            switch(addr) {

            case 0x00:

            case 0x01:

            case 0x02:

            case 0x03:

            case 0x06:

            case 0x07:

            case 0x08:

            case 0x09:

            case 0x0a:

            case 0x0b:

            case 0x0e:

            case 0x10 ... 0x27: /* base */

            case 0x2c ... 0x2f: /* read-only subsystem ID & vendor ID */

            case 0x30 ... 0x33: /* rom */

            case 0x3d:

                can_write = 0;

                break;

            default:

                can_write = 1;

                break;

            }

            break;

        default:

        case 0x01:

            switch(addr) {

            case 0x00:

            case 0x01:

            case 0x02:

            case 0x03:

            case 0x06:

            case 0x07:

            case 0x08:

            case 0x09:

            case 0x0a:

            case 0x0b:

            case 0x0e:

            case 0x2c ... 0x2f: /* read-only subsystem ID & vendor ID */

            case 0x38 ... 0x3b: /* rom */

            case 0x3d:

                can_write = 0;

                break;

            default:

                can_write = 1;

                break;

            }

            break;

        }

        if (can_write) {

            /* Mask out writes to reserved bits in registers */

            switch (addr) {

	    case 0x05:

                val &= ~PCI_COMMAND_RESERVED_MASK_HI;

                break;

            case 0x06:

                val &= ~PCI_STATUS_RESERVED_MASK_LO;

                break;

            case 0x07:

                val &= ~PCI_STATUS_RESERVED_MASK_HI;

                break;

            }

            d->config[addr] = val;

        }

        if (++addr > 0xff)

        	break;

        val >>= 8;

    }



    end = address + len;

    if (end > PCI_COMMAND && address < (PCI_COMMAND + 2)) {

        /* if the command register is modified, we must modify the mappings */

        pci_update_mappings(d);

    }

}
