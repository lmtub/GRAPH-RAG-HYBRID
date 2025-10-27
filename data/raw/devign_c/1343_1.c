static void esp_mem_writeb(void *opaque, target_phys_addr_t addr, uint32_t val)

{

    ESPState *s = opaque;

    uint32_t saddr;



    saddr = (addr >> s->it_shift) & (ESP_REGS - 1);

    DPRINTF("write reg[%d]: 0x%2.2x -> 0x%2.2x\n", saddr, s->wregs[saddr],

            val);

    switch (saddr) {

    case ESP_TCLO:

    case ESP_TCMID:

        s->rregs[ESP_RSTAT] &= ~STAT_TC;

        break;

    case ESP_FIFO:

        if (s->do_cmd) {

            s->cmdbuf[s->cmdlen++] = val & 0xff;

        } else if ((s->rregs[ESP_RSTAT] & STAT_PIO_MASK) == 0) {

            uint8_t buf;

            buf = val & 0xff;

            s->ti_size--;

            fprintf(stderr, "esp: PIO data write not implemented\n");

        } else {

            s->ti_size++;

            s->ti_buf[s->ti_wptr++] = val & 0xff;

        }

        break;

    case ESP_CMD:

        s->rregs[saddr] = val;

        if (val & CMD_DMA) {

            s->dma = 1;

            /* Reload DMA counter.  */

            s->rregs[ESP_TCLO] = s->wregs[ESP_TCLO];

            s->rregs[ESP_TCMID] = s->wregs[ESP_TCMID];

        } else {

            s->dma = 0;

        }

        switch(val & CMD_CMD) {

        case CMD_NOP:

            DPRINTF("NOP (%2.2x)\n", val);

            break;

        case CMD_FLUSH:

            DPRINTF("Flush FIFO (%2.2x)\n", val);

            //s->ti_size = 0;

            s->rregs[ESP_RINTR] = INTR_FC;

            s->rregs[ESP_RSEQ] = 0;

            s->rregs[ESP_RFLAGS] = 0;

            break;

        case CMD_RESET:

            DPRINTF("Chip reset (%2.2x)\n", val);

            esp_reset(s);

            break;

        case CMD_BUSRESET:

            DPRINTF("Bus reset (%2.2x)\n", val);

            s->rregs[ESP_RINTR] = INTR_RST;

            if (!(s->wregs[ESP_CFG1] & CFG1_RESREPT)) {

                esp_raise_irq(s);

            }

            break;

        case CMD_TI:

            handle_ti(s);

            break;

        case CMD_ICCS:

            DPRINTF("Initiator Command Complete Sequence (%2.2x)\n", val);

            write_response(s);

            break;

        case CMD_MSGACC:

            DPRINTF("Message Accepted (%2.2x)\n", val);

            write_response(s);

            s->rregs[ESP_RINTR] = INTR_DC;

            s->rregs[ESP_RSEQ] = 0;

            break;

        case CMD_SATN:

            DPRINTF("Set ATN (%2.2x)\n", val);

            break;

        case CMD_SELATN:

            DPRINTF("Set ATN (%2.2x)\n", val);

            handle_satn(s);

            break;

        case CMD_SELATNS:

            DPRINTF("Set ATN & stop (%2.2x)\n", val);

            handle_satn_stop(s);

            break;

        case CMD_ENSEL:

            DPRINTF("Enable selection (%2.2x)\n", val);

            break;

        default:

            DPRINTF("Unhandled ESP command (%2.2x)\n", val);

            break;

        }

        break;

    case ESP_WBUSID ... ESP_WSYNO:

        break;

    case ESP_CFG1:

        s->rregs[saddr] = val;

        break;

    case ESP_WCCF ... ESP_WTEST:

        break;

    case ESP_CFG2:

        s->rregs[saddr] = val & CFG2_MASK;

        break;

    case ESP_CFG3 ... ESP_RES4:

        s->rregs[saddr] = val;

        break;

    default:

        break;

    }

    s->wregs[saddr] = val;

}
