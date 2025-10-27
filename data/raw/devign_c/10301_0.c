static int zipl_magic(uint8_t *ptr)

{

    uint32_t *p = (void*)ptr;

    uint32_t *z = (void*)ZIPL_MAGIC;



    if (*p != *z) {

        debug_print_int("invalid magic", *p);

        virtio_panic("invalid magic");

    }



    return 1;

}
