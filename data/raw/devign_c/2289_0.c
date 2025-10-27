static int bochs_open(BlockDriverState *bs, const char *filename, int flags)

{

    BDRVBochsState *s = bs->opaque;

    int fd, i;

    struct bochs_header bochs;

    struct bochs_header_v1 header_v1;



    fd = open(filename, O_RDWR | O_BINARY);

    if (fd < 0) {

        fd = open(filename, O_RDONLY | O_BINARY);

        if (fd < 0)

            return -1;

    }



    bs->read_only = 1; // no write support yet



    s->fd = fd;



    if (read(fd, &bochs, sizeof(bochs)) != sizeof(bochs)) {

        goto fail;

    }



    if (strcmp(bochs.magic, HEADER_MAGIC) ||

        strcmp(bochs.type, REDOLOG_TYPE) ||

        strcmp(bochs.subtype, GROWING_TYPE) ||

	((le32_to_cpu(bochs.version) != HEADER_VERSION) &&

	(le32_to_cpu(bochs.version) != HEADER_V1))) {

        goto fail;

    }



    if (le32_to_cpu(bochs.version) == HEADER_V1) {

      memcpy(&header_v1, &bochs, sizeof(bochs));

      bs->total_sectors = le64_to_cpu(header_v1.extra.redolog.disk) / 512;

    } else {

      bs->total_sectors = le64_to_cpu(bochs.extra.redolog.disk) / 512;

    }



    lseek(s->fd, le32_to_cpu(bochs.header), SEEK_SET);



    s->catalog_size = le32_to_cpu(bochs.extra.redolog.catalog);

    s->catalog_bitmap = qemu_malloc(s->catalog_size * 4);

    if (read(s->fd, s->catalog_bitmap, s->catalog_size * 4) !=

	s->catalog_size * 4)

	goto fail;

    for (i = 0; i < s->catalog_size; i++)

	le32_to_cpus(&s->catalog_bitmap[i]);



    s->data_offset = le32_to_cpu(bochs.header) + (s->catalog_size * 4);



    s->bitmap_blocks = 1 + (le32_to_cpu(bochs.extra.redolog.bitmap) - 1) / 512;

    s->extent_blocks = 1 + (le32_to_cpu(bochs.extra.redolog.extent) - 1) / 512;



    s->extent_size = le32_to_cpu(bochs.extra.redolog.extent);



    return 0;

 fail:

    close(fd);

    return -1;

}
