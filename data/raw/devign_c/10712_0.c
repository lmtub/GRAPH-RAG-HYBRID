static uint64_t mv88w8618_flashcfg_read(void *opaque,

                                        target_phys_addr_t offset,

                                        unsigned size)

{

    mv88w8618_flashcfg_state *s = opaque;



    switch (offset) {

    case MP_FLASHCFG_CFGR0:

        return s->cfgr0;



    default:

        return 0;

    }

}
