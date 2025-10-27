static void qxl_log_image(PCIQXLDevice *qxl, QXLPHYSICAL addr, int group_id)

{

    QXLImage *image;

    QXLImageDescriptor *desc;



    image = qxl_phys2virt(qxl, addr, group_id);

    desc = &image->descriptor;

    fprintf(stderr, " (id %" PRIx64 " type %d flags %d width %d height %d",

            desc->id, desc->type, desc->flags, desc->width, desc->height);

    switch (desc->type) {

    case SPICE_IMAGE_TYPE_BITMAP:

        fprintf(stderr, ", fmt %d flags %d x %d y %d stride %d"

                " palette %" PRIx64 " data %" PRIx64,

                image->bitmap.format, image->bitmap.flags,

                image->bitmap.x, image->bitmap.y,

                image->bitmap.stride,

                image->bitmap.palette, image->bitmap.data);

        break;

    }

    fprintf(stderr, ")");

}
