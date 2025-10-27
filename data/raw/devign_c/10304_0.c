static void port92_write(void *opaque, hwaddr addr, uint64_t val,

                         unsigned size)

{

    Port92State *s = opaque;

    int oldval = s->outport;



    DPRINTF("port92: write 0x%02" PRIx64 "\n", val);

    s->outport = val;

    qemu_set_irq(*s->a20_out, (val >> 1) & 1);

    if ((val & 1) && !(oldval & 1)) {

        qemu_system_reset_request();

    }

}
