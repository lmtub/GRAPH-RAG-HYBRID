static int block_load(QEMUFile *f, void *opaque, int version_id)

{

    static int banner_printed;

    int len, flags;

    char device_name[256];

    int64_t addr;

    BlockDriverState *bs;

    uint8_t *buf;



    do {

        addr = qemu_get_be64(f);



        flags = addr & ~BDRV_SECTOR_MASK;

        addr >>= BDRV_SECTOR_BITS;



        if (flags & BLK_MIG_FLAG_DEVICE_BLOCK) {

            int ret;

            /* get device name */

            len = qemu_get_byte(f);

            qemu_get_buffer(f, (uint8_t *)device_name, len);

            device_name[len] = '\0';



            bs = bdrv_find(device_name);

            if (!bs) {

                fprintf(stderr, "Error unknown block device %s\n",

                        device_name);

                return -EINVAL;

            }



            buf = qemu_malloc(BLOCK_SIZE);



            qemu_get_buffer(f, buf, BLOCK_SIZE);

            ret = bdrv_write(bs, addr, buf, BDRV_SECTORS_PER_DIRTY_CHUNK);



            qemu_free(buf);

            if (ret < 0) {

                return ret;

            }

        } else if (flags & BLK_MIG_FLAG_PROGRESS) {

            if (!banner_printed) {

                printf("Receiving block device images\n");

                banner_printed = 1;

            }

            printf("Completed %d %%%c", (int)addr,

                   (addr == 100) ? '\n' : '\r');

            fflush(stdout);

        } else if (!(flags & BLK_MIG_FLAG_EOS)) {

            fprintf(stderr, "Unknown flags\n");

            return -EINVAL;

        }

        if (qemu_file_has_error(f)) {

            return -EIO;

        }

    } while (!(flags & BLK_MIG_FLAG_EOS));



    return 0;

}
