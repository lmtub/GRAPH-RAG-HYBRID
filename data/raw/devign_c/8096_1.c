parallel_ioport_write_sw(void *opaque, uint32_t addr, uint32_t val)

{

    ParallelState *s = opaque;



    pdebug("write addr=0x%02x val=0x%02x\n", addr, val);



    addr &= 7;

    switch(addr) {

    case PARA_REG_DATA:

        s->dataw = val;

        parallel_update_irq(s);

        break;

    case PARA_REG_CTR:

        val |= 0xc0;

        if ((val & PARA_CTR_INIT) == 0 ) {

            s->status = PARA_STS_BUSY;

            s->status |= PARA_STS_ACK;

            s->status |= PARA_STS_ONLINE;

            s->status |= PARA_STS_ERROR;

        }

        else if (val & PARA_CTR_SELECT) {

            if (val & PARA_CTR_STROBE) {

                s->status &= ~PARA_STS_BUSY;

                if ((s->control & PARA_CTR_STROBE) == 0)

                    qemu_chr_fe_write(s->chr, &s->dataw, 1);

            } else {

                if (s->control & PARA_CTR_INTEN) {

                    s->irq_pending = 1;

                }

            }

        }

        parallel_update_irq(s);

        s->control = val;

        break;

    }

}
