static uint64_t escc_mem_read(void *opaque, target_phys_addr_t addr,

                              unsigned size)

{

    SerialState *serial = opaque;

    ChannelState *s;

    uint32_t saddr;

    uint32_t ret;

    int channel;



    saddr = (addr >> serial->it_shift) & 1;

    channel = (addr >> (serial->it_shift + 1)) & 1;

    s = &serial->chn[channel];

    switch (saddr) {

    case SERIAL_CTRL:

        trace_escc_mem_readb_ctrl(CHN_C(s), s->reg, s->rregs[s->reg]);

        ret = s->rregs[s->reg];

        s->reg = 0;

        return ret;

    case SERIAL_DATA:

        s->rregs[R_STATUS] &= ~STATUS_RXAV;

        clr_rxint(s);

        if (s->type == kbd || s->type == mouse)

            ret = get_queue(s);

        else

            ret = s->rx;

        trace_escc_mem_readb_data(CHN_C(s), ret);

        if (s->chr)

            qemu_chr_accept_input(s->chr);

        return ret;

    default:

        break;

    }

    return 0;

}
