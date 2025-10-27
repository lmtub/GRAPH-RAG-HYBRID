int load_aout(const char *filename, target_phys_addr_t addr, int max_sz,

              int bswap_needed, target_phys_addr_t target_page_size)

{

    int fd, size, ret;

    struct exec e;

    uint32_t magic;



    fd = open(filename, O_RDONLY | O_BINARY);

    if (fd < 0)

        return -1;



    size = read(fd, &e, sizeof(e));

    if (size < 0)

        goto fail;



    if (bswap_needed) {

        bswap_ahdr(&e);

    }



    magic = N_MAGIC(e);

    switch (magic) {

    case ZMAGIC:

    case QMAGIC:

    case OMAGIC:

        if (e.a_text + e.a_data > max_sz)

            goto fail;

	lseek(fd, N_TXTOFF(e), SEEK_SET);

	size = read_targphys(filename, fd, addr, e.a_text + e.a_data);

	if (size < 0)

	    goto fail;

	break;

    case NMAGIC:

        if (N_DATADDR(e, target_page_size) + e.a_data > max_sz)

            goto fail;

	lseek(fd, N_TXTOFF(e), SEEK_SET);

	size = read_targphys(filename, fd, addr, e.a_text);

	if (size < 0)

	    goto fail;

        ret = read_targphys(filename, fd, addr + N_DATADDR(e, target_page_size),

                            e.a_data);

	if (ret < 0)

	    goto fail;

	size += ret;

	break;

    default:

	goto fail;

    }

    close(fd);

    return size;

 fail:

    close(fd);

    return -1;

}
