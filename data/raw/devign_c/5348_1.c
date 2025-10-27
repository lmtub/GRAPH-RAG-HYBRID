static int guess_disk_lchs(IDEState *s,

                           int *pcylinders, int *pheads, int *psectors)

{

    uint8_t *buf;

    int ret, i, heads, sectors, cylinders;

    struct partition *p;

    uint32_t nr_sects;



    buf = qemu_memalign(512, 512);

    if (buf == NULL)

        return -1;

    ret = bdrv_read(s->bs, 0, buf, 1);

    if (ret < 0) {

        qemu_free(buf);

        return -1;

    }

    /* test msdos magic */

    if (buf[510] != 0x55 || buf[511] != 0xaa) {

        qemu_free(buf);

        return -1;

    }

    for(i = 0; i < 4; i++) {

        p = ((struct partition *)(buf + 0x1be)) + i;

        nr_sects = le32_to_cpu(p->nr_sects);

        if (nr_sects && p->end_head) {

            /* We make the assumption that the partition terminates on

               a cylinder boundary */

            heads = p->end_head + 1;

            sectors = p->end_sector & 63;

            if (sectors == 0)

                continue;

            cylinders = s->nb_sectors / (heads * sectors);

            if (cylinders < 1 || cylinders > 16383)

                continue;

            *pheads = heads;

            *psectors = sectors;

            *pcylinders = cylinders;

#if 0

            printf("guessed geometry: LCHS=%d %d %d\n",

                   cylinders, heads, sectors);

#endif

            qemu_free(buf);

            return 0;

        }

    }

    qemu_free(buf);

    return -1;

}
