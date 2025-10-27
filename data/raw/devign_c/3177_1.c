unsigned long virtio_load_direct(ulong rec_list1, ulong rec_list2,

                                 ulong subchan_id, void *load_addr)

{

    u8 status;

    int sec = rec_list1;

    int sec_num = ((rec_list2 >> 32) & 0xffff) + 1;

    int sec_len = rec_list2 >> 48;

    ulong addr = (ulong)load_addr;



    if (sec_len != virtio_get_block_size()) {

        return -1;

    }



    sclp_print(".");

    status = virtio_read_many(sec, (void *)addr, sec_num);

    if (status) {

        virtio_panic("I/O Error");

    }

    addr += sec_num * virtio_get_block_size();



    return addr;

}
