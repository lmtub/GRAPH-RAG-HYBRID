static int ram_load_dead(QEMUFile *f, void *opaque)

{

    RamDecompressState s1, *s = &s1;

    uint8_t buf[10];

    ram_addr_t i;



    if (ram_decompress_open(s, f) < 0)

        return -EINVAL;

    for(i = 0; i < last_ram_offset; i+= BDRV_HASH_BLOCK_SIZE) {

        if (ram_decompress_buf(s, buf, 1) < 0) {

            fprintf(stderr, "Error while reading ram block header\n");

            goto error;

        }

        if (buf[0] == 0) {

            if (ram_decompress_buf(s, qemu_get_ram_ptr(i),

                                   BDRV_HASH_BLOCK_SIZE) < 0) {

                fprintf(stderr, "Error while reading ram block address=0x%08" PRIx64, (uint64_t)i);

                goto error;

            }

        } else {

        error:

            printf("Error block header\n");

            return -EINVAL;

        }

    }

    ram_decompress_close(s);



    return 0;

}
