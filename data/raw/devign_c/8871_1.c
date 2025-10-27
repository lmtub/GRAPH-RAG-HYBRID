static void qxl_exit_vga_mode(PCIQXLDevice *d)

{

    if (d->mode != QXL_MODE_VGA) {

        return;

    }

    trace_qxl_exit_vga_mode(d->id);


    qxl_destroy_primary(d, QXL_SYNC);

}