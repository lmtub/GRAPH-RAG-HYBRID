static void qemu_spice_display_init_one(QemuConsole *con)

{

    SimpleSpiceDisplay *ssd = g_new0(SimpleSpiceDisplay, 1);



    qemu_spice_display_init_common(ssd);



    ssd->qxl.base.sif = &dpy_interface.base;

    qemu_spice_add_display_interface(&ssd->qxl, con);

    assert(ssd->worker);



    qemu_spice_create_host_memslot(ssd);



    ssd->dcl.ops = &display_listener_ops;

    ssd->dcl.con = con;

    register_displaychangelistener(&ssd->dcl);

}
