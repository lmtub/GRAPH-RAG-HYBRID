static uint32_t vbe_ioport_read_index(void *opaque, uint32_t addr)

{

    VGACommonState *s = opaque;

    uint32_t val;

    val = s->vbe_index;

    return val;

}
