MacIONVRAMState *macio_nvram_init (target_phys_addr_t size,

                                   unsigned int it_shift)

{

    MacIONVRAMState *s;



    s = g_malloc0(sizeof(MacIONVRAMState));

    s->data = g_malloc0(size);

    s->size = size;

    s->it_shift = it_shift;



    memory_region_init_io(&s->mem, &macio_nvram_ops, s, "macio-nvram",

                          size << it_shift);

    vmstate_register(NULL, -1, &vmstate_macio_nvram, s);

    qemu_register_reset(macio_nvram_reset, s);



    return s;

}
