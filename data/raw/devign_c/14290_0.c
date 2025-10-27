static void device_set_bootindex(Object *obj, Visitor *v, const char *name,

                                 void *opaque, Error **errp)

{

    BootIndexProperty *prop = opaque;

    int32_t boot_index;

    Error *local_err = NULL;



    visit_type_int32(v, name, &boot_index, &local_err);

    if (local_err) {

        goto out;

    }

    /* check whether bootindex is present in fw_boot_order list  */

    check_boot_index(boot_index, &local_err);

    if (local_err) {

        goto out;

    }

    /* change bootindex to a new one */

    *prop->bootindex = boot_index;



    add_boot_device_path(*prop->bootindex, prop->dev, prop->suffix);



out:

    if (local_err) {

        error_propagate(errp, local_err);

    }

}
