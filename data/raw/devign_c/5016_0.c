static bool eckd_valid_address(BootMapPointer *p)

{

    const uint64_t cylinder = p->eckd.cylinder

                            + ((p->eckd.head & 0xfff0) << 12);

    const uint64_t head = p->eckd.head & 0x000f;



    if (head >= virtio_get_heads()

        ||  p->eckd.sector > virtio_get_sectors()

        ||  p->eckd.sector <= 0) {

        return false;

    }



    if (!virtio_guessed_disk_nature() && cylinder >= virtio_get_cylinders()) {

        return false;

    }



    return true;

}
