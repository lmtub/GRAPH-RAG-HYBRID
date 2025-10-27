static block_number_t eckd_block_num(BootMapPointer *p)

{

    const uint64_t sectors = virtio_get_sectors();

    const uint64_t heads = virtio_get_heads();

    const uint64_t cylinder = p->eckd.cylinder

                            + ((p->eckd.head & 0xfff0) << 12);

    const uint64_t head = p->eckd.head & 0x000f;

    const block_number_t block = sectors * heads * cylinder

                               + sectors * head

                               + p->eckd.sector

                               - 1; /* block nr starts with zero */

    return block;

}
