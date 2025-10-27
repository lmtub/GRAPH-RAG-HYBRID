static void sh_serial_write(void *opaque, hwaddr offs,

                            uint64_t val, unsigned size)

{

    sh_serial_state *s = opaque;

    unsigned char ch;



#ifdef DEBUG_SERIAL

    printf("sh_serial: write offs=0x%02x val=0x%02x\n",

	   offs, val);

#endif

    switch(offs) {

    case 0x00: /* SMR */

        s->smr = val & ((s->feat & SH_SERIAL_FEAT_SCIF) ? 0x7b : 0xff);

        return;

    case 0x04: /* BRR */

        s->brr = val;

	return;

    case 0x08: /* SCR */

        /* TODO : For SH7751, SCIF mask should be 0xfb. */

        s->scr = val & ((s->feat & SH_SERIAL_FEAT_SCIF) ? 0xfa : 0xff);

        if (!(val & (1 << 5)))

            s->flags |= SH_SERIAL_FLAG_TEND;

        if ((s->feat & SH_SERIAL_FEAT_SCIF) && s->txi) {

	    qemu_set_irq(s->txi, val & (1 << 7));

        }

        if (!(val & (1 << 6))) {

	    qemu_set_irq(s->rxi, 0);

        }

        return;

    case 0x0c: /* FTDR / TDR */

        if (s->chr) {

            ch = val;

            qemu_chr_fe_write(s->chr, &ch, 1);

	}

	s->dr = val;

	s->flags &= ~SH_SERIAL_FLAG_TDE;

        return;

#if 0

    case 0x14: /* FRDR / RDR */

        ret = 0;

        break;

#endif

    }

    if (s->feat & SH_SERIAL_FEAT_SCIF) {

        switch(offs) {

        case 0x10: /* FSR */

            if (!(val & (1 << 6)))

                s->flags &= ~SH_SERIAL_FLAG_TEND;

            if (!(val & (1 << 5)))

                s->flags &= ~SH_SERIAL_FLAG_TDE;

            if (!(val & (1 << 4)))

                s->flags &= ~SH_SERIAL_FLAG_BRK;

            if (!(val & (1 << 1)))

                s->flags &= ~SH_SERIAL_FLAG_RDF;

            if (!(val & (1 << 0)))

                s->flags &= ~SH_SERIAL_FLAG_DR;



            if (!(val & (1 << 1)) || !(val & (1 << 0))) {

                if (s->rxi) {

                    qemu_set_irq(s->rxi, 0);

                }

            }

            return;

        case 0x18: /* FCR */

            s->fcr = val;

            switch ((val >> 6) & 3) {

            case 0:

                s->rtrg = 1;

                break;

            case 1:

                s->rtrg = 4;

                break;

            case 2:

                s->rtrg = 8;

                break;

            case 3:

                s->rtrg = 14;

                break;

            }

            if (val & (1 << 1)) {

                sh_serial_clear_fifo(s);

                s->sr &= ~(1 << 1);

            }



            return;

        case 0x20: /* SPTR */

            s->sptr = val & 0xf3;

            return;

        case 0x24: /* LSR */

            return;

        }

    }

    else {

        switch(offs) {

#if 0

        case 0x0c:

            ret = s->dr;

            break;

        case 0x10:

            ret = 0;

            break;

#endif

        case 0x1c:

            s->sptr = val & 0x8f;

            return;

        }

    }



    fprintf(stderr, "sh_serial: unsupported write to 0x%02"

            HWADDR_PRIx "\n", offs);

    abort();

}
