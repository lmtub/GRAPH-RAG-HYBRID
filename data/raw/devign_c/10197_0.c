ioapic_mem_read(void *opaque, target_phys_addr_t addr, unsigned int size)

{

    IOAPICCommonState *s = opaque;

    int index;

    uint32_t val = 0;



    switch (addr & 0xff) {

    case IOAPIC_IOREGSEL:

        val = s->ioregsel;

        break;

    case IOAPIC_IOWIN:

        if (size != 4) {

            break;

        }

        switch (s->ioregsel) {

        case IOAPIC_REG_ID:

            val = s->id << IOAPIC_ID_SHIFT;

            break;

        case IOAPIC_REG_VER:

            val = IOAPIC_VERSION |

                ((IOAPIC_NUM_PINS - 1) << IOAPIC_VER_ENTRIES_SHIFT);

            break;

        case IOAPIC_REG_ARB:

            val = 0;

            break;

        default:

            index = (s->ioregsel - IOAPIC_REG_REDTBL_BASE) >> 1;

            if (index >= 0 && index < IOAPIC_NUM_PINS) {

                if (s->ioregsel & 1) {

                    val = s->ioredtbl[index] >> 32;

                } else {

                    val = s->ioredtbl[index] & 0xffffffff;

                }

            }

        }

        DPRINTF("read: %08x = %08x\n", s->ioregsel, val);

        break;

    }

    return val;

}
