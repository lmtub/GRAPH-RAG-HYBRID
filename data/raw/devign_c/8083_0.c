static void usb_msd_set_bootindex(Object *obj, Visitor *v, const char *name,

                                  void *opaque, Error **errp)

{

    USBDevice *dev = USB_DEVICE(obj);

    MSDState *s = USB_STORAGE_DEV(dev);

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

    s->conf.bootindex = boot_index;



    if (s->scsi_dev) {

        object_property_set_int(OBJECT(s->scsi_dev), boot_index, "bootindex",

                                &error_abort);

    }



out:

    if (local_err) {

        error_propagate(errp, local_err);

    }

}
