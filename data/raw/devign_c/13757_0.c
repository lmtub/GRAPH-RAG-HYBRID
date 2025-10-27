static void mv88w8618_pic_write(void *opaque, target_phys_addr_t offset,

                                uint64_t value, unsigned size)

{

    mv88w8618_pic_state *s = opaque;



    switch (offset) {

    case MP_PIC_ENABLE_SET:

        s->enabled |= value;

        break;



    case MP_PIC_ENABLE_CLR:

        s->enabled &= ~value;

        s->level &= ~value;

        break;

    }

    mv88w8618_pic_update(s);

}
