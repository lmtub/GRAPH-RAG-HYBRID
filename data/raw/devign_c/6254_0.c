build_append_notify(GArray *device, const char *name,

                    const char *format, int skip, int count)

{

    int i;

    GArray *method = build_alloc_array();

    uint8_t op = 0x14; /* MethodOp */



    build_append_nameseg(method, "%s", name);

    build_append_byte(method, 0x02); /* MethodFlags: ArgCount */

    for (i = skip; i < count; i++) {

        GArray *target = build_alloc_array();

        build_append_nameseg(target, format, i);

        assert(i < 256); /* Fits in 1 byte */

        build_append_notify_target(method, target, i, 1);

        build_free_array(target);

    }

    build_package(method, op, 2);



    build_append_array(device, method);

    build_free_array(method);

}
