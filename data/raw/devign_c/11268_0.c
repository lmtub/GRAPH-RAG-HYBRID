static int create_dynamic_disk(int fd, uint8_t *buf, int64_t total_sectors)

{

    VHDDynDiskHeader *dyndisk_header =

        (VHDDynDiskHeader *) buf;

    size_t block_size, num_bat_entries;

    int i;

    int ret = -EIO;



    // Write the footer (twice: at the beginning and at the end)

    block_size = 0x200000;

    num_bat_entries = (total_sectors + block_size / 512) / (block_size / 512);



    if (write(fd, buf, HEADER_SIZE) != HEADER_SIZE) {

        goto fail;

    }



    if (lseek(fd, 1536 + ((num_bat_entries * 4 + 511) & ~511), SEEK_SET) < 0) {

        goto fail;

    }

    if (write(fd, buf, HEADER_SIZE) != HEADER_SIZE) {

        goto fail;

    }



    // Write the initial BAT

    if (lseek(fd, 3 * 512, SEEK_SET) < 0) {

        goto fail;

    }



    memset(buf, 0xFF, 512);

    for (i = 0; i < (num_bat_entries * 4 + 511) / 512; i++) {

        if (write(fd, buf, 512) != 512) {

            goto fail;

        }

    }



    // Prepare the Dynamic Disk Header

    memset(buf, 0, 1024);



    memcpy(dyndisk_header->magic, "cxsparse", 8);



    /*

     * Note: The spec is actually wrong here for data_offset, it says

     * 0xFFFFFFFF, but MS tools expect all 64 bits to be set.

     */

    dyndisk_header->data_offset = be64_to_cpu(0xFFFFFFFFFFFFFFFFULL);

    dyndisk_header->table_offset = be64_to_cpu(3 * 512);

    dyndisk_header->version = be32_to_cpu(0x00010000);

    dyndisk_header->block_size = be32_to_cpu(block_size);

    dyndisk_header->max_table_entries = be32_to_cpu(num_bat_entries);



    dyndisk_header->checksum = be32_to_cpu(vpc_checksum(buf, 1024));



    // Write the header

    if (lseek(fd, 512, SEEK_SET) < 0) {

        goto fail;

    }



    if (write(fd, buf, 1024) != 1024) {

        goto fail;

    }

    ret = 0;



 fail:

    return ret;

}
