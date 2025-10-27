static void qemu_spice_create_update(SimpleSpiceDisplay *ssd)

{

    static const int blksize = 32;

    int blocks = (surface_width(ssd->ds) + blksize - 1) / blksize;

    int dirty_top[blocks];

    int y, yoff, x, xoff, blk, bw;

    int bpp = surface_bytes_per_pixel(ssd->ds);

    uint8_t *guest, *mirror;



    if (qemu_spice_rect_is_empty(&ssd->dirty)) {

        return;

    };



    for (blk = 0; blk < blocks; blk++) {

        dirty_top[blk] = -1;

    }



    guest = surface_data(ssd->ds);

    mirror = (void *)pixman_image_get_data(ssd->mirror);

    for (y = ssd->dirty.top; y < ssd->dirty.bottom; y++) {

        yoff = y * surface_stride(ssd->ds);

        for (x = ssd->dirty.left; x < ssd->dirty.right; x += blksize) {

            xoff = x * bpp;

            blk = x / blksize;

            bw = MIN(blksize, ssd->dirty.right - x);

            if (memcmp(guest + yoff + xoff,

                       mirror + yoff + xoff,

                       bw * bpp) == 0) {

                if (dirty_top[blk] != -1) {

                    QXLRect update = {

                        .top    = dirty_top[blk],

                        .bottom = y,

                        .left   = x,

                        .right  = x + bw,

                    };

                    qemu_spice_create_one_update(ssd, &update);

                    dirty_top[blk] = -1;

                }

            } else {

                if (dirty_top[blk] == -1) {

                    dirty_top[blk] = y;

                }

            }

        }

    }



    for (x = ssd->dirty.left; x < ssd->dirty.right; x += blksize) {

        blk = x / blksize;

        bw = MIN(blksize, ssd->dirty.right - x);

        if (dirty_top[blk] != -1) {

            QXLRect update = {

                .top    = dirty_top[blk],

                .bottom = ssd->dirty.bottom,

                .left   = x,

                .right  = x + bw,

            };

            qemu_spice_create_one_update(ssd, &update);

            dirty_top[blk] = -1;

        }

    }



    memset(&ssd->dirty, 0, sizeof(ssd->dirty));

}
