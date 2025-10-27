static int zipl_run(struct scsi_blockptr *pte)

{

    struct component_header *header;

    struct component_entry *entry;

    uint8_t tmp_sec[SECTOR_SIZE];



    virtio_read(pte->blockno, tmp_sec);

    header = (struct component_header *)tmp_sec;



    if (!zipl_magic(tmp_sec)) {

        goto fail;

    }



    if (header->type != ZIPL_COMP_HEADER_IPL) {

        goto fail;

    }



    dputs("start loading images\n");



    /* Load image(s) into RAM */

    entry = (struct component_entry *)(&header[1]);

    while (entry->component_type == ZIPL_COMP_ENTRY_LOAD) {

        if (zipl_load_segment(entry) < 0) {

            goto fail;

        }



        entry++;



        if ((uint8_t*)(&entry[1]) > (tmp_sec + SECTOR_SIZE)) {

            goto fail;

        }

    }



    if (entry->component_type != ZIPL_COMP_ENTRY_EXEC) {

        goto fail;

    }



    /* should not return */

    jump_to_IPL_code(entry->load_address);



    return 0;



fail:

    sclp_print("failed running zipl\n");

    return -1;

}
