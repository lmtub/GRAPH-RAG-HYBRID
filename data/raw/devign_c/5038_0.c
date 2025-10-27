static uint64_t mv88w8618_pic_read(void *opaque, target_phys_addr_t offset,

                                   unsigned size)

{

    mv88w8618_pic_state *s = opaque;



    switch (offset) {

    case MP_PIC_STATUS:

        return s->level & s->enabled;



    default:

        return 0;

    }

}
