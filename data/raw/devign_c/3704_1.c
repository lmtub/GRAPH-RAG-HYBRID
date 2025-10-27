e1000e_io_read(void *opaque, hwaddr addr, unsigned size)

{

    E1000EState *s = opaque;

    uint32_t idx;

    uint64_t val;



    switch (addr) {

    case E1000_IOADDR:

        trace_e1000e_io_read_addr(s->ioaddr);

        return s->ioaddr;

    case E1000_IODATA:

        if (e1000e_io_get_reg_index(s, &idx)) {

            val = e1000e_core_read(&s->core, idx, sizeof(val));

            trace_e1000e_io_read_data(idx, val);

            return val;

        }

        return 0;

    default:

        trace_e1000e_wrn_io_read_unknown(addr);

        return 0;

    }

}
