static void pci_realview_class_init(ObjectClass *class, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(class);



    /* Reason: object_unref() hangs */

    dc->cannot_destroy_with_object_finalize_yet = true;

}
