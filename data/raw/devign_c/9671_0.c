static void qxl_hw_screen_dump(void *opaque, const char *filename, bool cswitch,

                               Error **errp)

{

    PCIQXLDevice *qxl = opaque;

    VGACommonState *vga = &qxl->vga;



    switch (qxl->mode) {

    case QXL_MODE_COMPAT:

    case QXL_MODE_NATIVE:

        qxl_render_update(qxl);

        ppm_save(filename, qxl->ssd.ds, errp);

        break;

    case QXL_MODE_VGA:

        vga->screen_dump(vga, filename, cswitch, errp);

        break;

    default:

        break;

    }

}
