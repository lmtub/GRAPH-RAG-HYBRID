static void build_append_notify_target(GArray *method, GArray *target_name,

                                       uint32_t value, int size)

{

    GArray *notify = build_alloc_array();

    uint8_t op = 0xA0; /* IfOp */



    build_append_byte(notify, 0x93); /* LEqualOp */

    build_append_byte(notify, 0x68); /* Arg0Op */

    build_append_value(notify, value, size);

    build_append_byte(notify, 0x86); /* NotifyOp */

    build_append_array(notify, target_name);

    build_append_byte(notify, 0x69); /* Arg1Op */



    /* Pack it up */

    build_package(notify, op, 1);



    build_append_array(method, notify);



    build_free_array(notify);

}
