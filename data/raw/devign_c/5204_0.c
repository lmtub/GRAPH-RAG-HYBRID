static void ivshmem_plain_init(Object *obj)

{

    IVShmemState *s = IVSHMEM_PLAIN(obj);



    object_property_add_link(obj, "memdev", TYPE_MEMORY_BACKEND,

                             (Object **)&s->hostmem,

                             ivshmem_check_memdev_is_busy,

                             OBJ_PROP_LINK_UNREF_ON_RELEASE,

                             &error_abort);

    s->not_legacy_32bit = 1;

}
