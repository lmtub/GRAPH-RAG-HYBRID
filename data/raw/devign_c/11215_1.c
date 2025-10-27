e1000e_io_write(void *opaque, hwaddr addr,

                uint64_t val, unsigned size)

{

    E1000EState *s = opaque;

    uint32_t idx;



    switch (addr) {

    case E1000_IOADDR:

        trace_e1000e_io_write_addr(val);

        s->ioaddr = (uint32_t) val;

        return;

    case E1000_IODATA:

        if (e1000e_io_get_reg_index(s, &idx)) {

            trace_e1000e_io_write_data(idx, val);

            e1000e_core_write(&s->core, idx, val, sizeof(val));

        }

        return;

    default:

        trace_e1000e_wrn_io_write_unknown(addr);

        return;

    }

}
