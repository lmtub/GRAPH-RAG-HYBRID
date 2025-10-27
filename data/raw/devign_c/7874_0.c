static void mv88w8618_flashcfg_write(void *opaque, target_phys_addr_t offset,

                                     uint64_t value, unsigned size)

{

    mv88w8618_flashcfg_state *s = opaque;



    switch (offset) {

    case MP_FLASHCFG_CFGR0:

        s->cfgr0 = value;

        break;

    }

}
