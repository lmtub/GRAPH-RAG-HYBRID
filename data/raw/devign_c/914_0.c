static void cs_write (void *opaque, target_phys_addr_t addr,

                      uint64_t val64, unsigned size)

{

    CSState *s = opaque;

    uint32_t saddr, iaddr, val;



    saddr = addr;

    val = val64;



    switch (saddr) {

    case Index_Address:

        if (!(s->regs[Index_Address] & MCE) && (val & MCE)

            && (s->dregs[Interface_Configuration] & (3 << 3)))

            s->aci_counter = conf.aci_counter;



        s->regs[Index_Address] = val & ~(1 << 7);

        break;



    case Index_Data:

        if (!(s->dregs[MODE_And_ID] & MODE2))

            iaddr = s->regs[Index_Address] & 0x0f;

        else

            iaddr = s->regs[Index_Address] & 0x1f;



        switch (iaddr) {

        case RESERVED:

        case RESERVED_2:

        case RESERVED_3:

            lwarn ("attempt to write %#x to reserved indirect register %d\n",

                   val, iaddr);

            break;



        case FS_And_Playback_Data_Format:

            if (s->regs[Index_Address] & MCE) {

                cs_reset_voices (s, val);

            }

            else {

                if (s->dregs[Alternate_Feature_Status] & PMCE) {

                    val = (val & ~0x0f) | (s->dregs[iaddr] & 0x0f);

                    cs_reset_voices (s, val);

                }

                else {

                    lwarn ("[P]MCE(%#x, %#x) is not set, val=%#x\n",

                           s->regs[Index_Address],

                           s->dregs[Alternate_Feature_Status],

                           val);

                    break;

                }

            }

            s->dregs[iaddr] = val;

            break;



        case Interface_Configuration:

            val &= ~(1 << 5);   /* D5 is reserved */

            s->dregs[iaddr] = val;

            if (val & PPIO) {

                lwarn ("PIO is not supported (%#x)\n", val);

                break;

            }

            if (val & PEN) {

                if (!s->dma_running) {

                    cs_reset_voices (s, s->dregs[FS_And_Playback_Data_Format]);

                }

            }

            else {

                if (s->dma_running) {

                    DMA_release_DREQ (s->dma);

                    AUD_set_active_out (s->voice, 0);

                    s->dma_running = 0;

                }

            }

            break;



        case Error_Status_And_Initialization:

            lwarn ("attempt to write to read only register %d\n", iaddr);

            break;



        case MODE_And_ID:

            dolog ("val=%#x\n", val);

            if (val & MODE2)

                s->dregs[iaddr] |= MODE2;

            else

                s->dregs[iaddr] &= ~MODE2;

            break;



        case Alternate_Feature_Enable_I:

            if (val & TE)

                lerr ("timer is not yet supported\n");

            s->dregs[iaddr] = val;

            break;



        case Alternate_Feature_Status:

            if ((s->dregs[iaddr] & PI) && !(val & PI)) {

                /* XXX: TI CI */

                qemu_irq_lower (s->pic);

                s->regs[Status] &= ~INT;

            }

            s->dregs[iaddr] = val;

            break;



        case Version_Chip_ID:

            lwarn ("write to Version_Chip_ID register %#x\n", val);

            s->dregs[iaddr] = val;

            break;



        default:

            s->dregs[iaddr] = val;

            break;

        }

        dolog ("written value %#x to indirect register %d\n", val, iaddr);

        break;



    case Status:

        if (s->regs[Status] & INT) {

            qemu_irq_lower (s->pic);

        }

        s->regs[Status] &= ~INT;

        s->dregs[Alternate_Feature_Status] &= ~(PI | CI | TI);

        break;



    case PIO_Data:

        lwarn ("attempt to write value %#x to PIO register\n", val);

        break;

    }

}
