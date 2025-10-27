static int zipl_load_segment(struct component_entry *entry)

{

    const int max_entries = (SECTOR_SIZE / sizeof(struct scsi_blockptr));

    struct scsi_blockptr *bprs = (void*)sec;

    const int bprs_size = sizeof(sec);

    uint64_t blockno;

    long address;

    int i;



    blockno = entry->data.blockno;

    address = entry->load_address;



    debug_print_int("loading segment at block", blockno);

    debug_print_int("addr", address);



    do {

        memset(bprs, FREE_SPACE_FILLER, bprs_size);

        if (virtio_read(blockno, (uint8_t *)bprs)) {

            debug_print_int("failed reading bprs at", blockno);

            goto fail;

        }



        for (i = 0;; i++) {

            u64 *cur_desc = (void*)&bprs[i];



            blockno = bprs[i].blockno;

            if (!blockno)

                break;



            /* we need the updated blockno for the next indirect entry in the

               chain, but don't want to advance address */

            if (i == (max_entries - 1))

                break;



            if (bprs[i].blockct == 0 && unused_space(&bprs[i + 1],

                sizeof(struct scsi_blockptr))) {

                /* This is a "continue" pointer.

                 * This ptr is the last one in the current script section.

                 * I.e. the next ptr must point to the unused memory area.

                 * The blockno is not zero, so the upper loop must continue

                 * reading next section of BPRS.

                 */

                break;

            }

            address = virtio_load_direct(cur_desc[0], cur_desc[1], 0,

                                         (void*)address);

            if (address == -1)

                goto fail;

        }

    } while (blockno);



    return 0;



fail:

    sclp_print("failed loading segment\n");

    return -1;

}
