static void virtio_setup(void)

{

    struct irb irb;

    int i;

    int r;

    bool found = false;



    blk_schid.one = 1;



    for (i = 0; i < 0x10000; i++) {

        blk_schid.sch_no = i;

        r = tsch(blk_schid, &irb);

        if (r != 3) {

            if (virtio_is_blk(blk_schid)) {

                found = true;

                break;

            }

        }

    }



    if (!found) {

        virtio_panic("No virtio-blk device found!\n");

    }



    virtio_setup_block(blk_schid);

}
