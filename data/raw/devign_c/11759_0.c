static void slavio_serial_mem_writeb(void *opaque, target_phys_addr_t addr, uint32_t val)

{

    SerialState *ser = opaque;

    ChannelState *s;

    uint32_t saddr;

    int newreg, channel;



    val &= 0xff;

    saddr = (addr & 3) >> 1;

    channel = (addr & SERIAL_MAXADDR) >> 2;

    s = &ser->chn[channel];

    switch (saddr) {

    case 0:

	SER_DPRINTF("Write channel %c, reg[%d] = %2.2x\n", CHN_C(s), s->reg, val & 0xff);

	newreg = 0;

	switch (s->reg) {

	case 0:

	    newreg = val & 7;

	    val &= 0x38;

	    switch (val) {

	    case 8:

		newreg |= 0x8;

		break;

	    case 0x28:

                clr_txint(s);

		break;

	    case 0x38:

                if (s->rxint_under_svc)

                    clr_rxint(s);

                else if (s->txint_under_svc)

                    clr_txint(s);

		break;

	    default:

		break;

	    }

	    break;

        case 1 ... 3:

        case 6 ... 8:

        case 10 ... 11:

        case 14 ... 15:

	    s->wregs[s->reg] = val;

	    break;

        case 4:

        case 5:

        case 12:

        case 13:

	    s->wregs[s->reg] = val;

            slavio_serial_update_parameters(s);

	    break;

	case 9:

	    switch (val & 0xc0) {

	    case 0:

	    default:

		break;

	    case 0x40:

		slavio_serial_reset_chn(&ser->chn[1]);

		return;

	    case 0x80:

		slavio_serial_reset_chn(&ser->chn[0]);

		return;

	    case 0xc0:

		slavio_serial_reset(ser);

		return;

	    }

	    break;

	default:

	    break;

	}

	if (s->reg == 0)

	    s->reg = newreg;

	else

	    s->reg = 0;

	break;

    case 1:

	SER_DPRINTF("Write channel %c, ch %d\n", CHN_C(s), val);

	if (s->wregs[5] & 8) { // tx enabled

	    s->tx = val;

	    if (s->chr)

		qemu_chr_write(s->chr, &s->tx, 1);

	    else if (s->type == kbd) {

		handle_kbd_command(s, val);

	    }

	    s->rregs[0] |= 4; // Tx buffer empty

	    s->rregs[1] |= 1; // All sent

            set_txint(s);

	}

	break;

    default:

	break;

    }

}
