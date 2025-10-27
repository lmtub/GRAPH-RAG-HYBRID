int load_uimage(const char *filename, target_phys_addr_t *ep,

                target_phys_addr_t *loadaddr, int *is_linux)

{

    int fd;

    int size;

    uboot_image_header_t h;

    uboot_image_header_t *hdr = &h;

    uint8_t *data = NULL;

    int ret = -1;



    fd = open(filename, O_RDONLY | O_BINARY);

    if (fd < 0)

        return -1;



    size = read(fd, hdr, sizeof(uboot_image_header_t));

    if (size < 0)

        goto out;



    bswap_uboot_header(hdr);



    if (hdr->ih_magic != IH_MAGIC)

        goto out;



    /* TODO: Implement other image types.  */

    if (hdr->ih_type != IH_TYPE_KERNEL) {

        fprintf(stderr, "Can only load u-boot image type \"kernel\"\n");

        goto out;

    }



    switch (hdr->ih_comp) {

    case IH_COMP_NONE:

    case IH_COMP_GZIP:

        break;

    default:

        fprintf(stderr,

                "Unable to load u-boot images with compression type %d\n",

                hdr->ih_comp);

        goto out;

    }



    /* TODO: Check CPU type.  */

    if (is_linux) {

        if (hdr->ih_os == IH_OS_LINUX)

            *is_linux = 1;

        else

            *is_linux = 0;

    }



    *ep = hdr->ih_ep;

    data = qemu_malloc(hdr->ih_size);



    if (read(fd, data, hdr->ih_size) != hdr->ih_size) {

        fprintf(stderr, "Error reading file\n");

        goto out;

    }



    if (hdr->ih_comp == IH_COMP_GZIP) {

        uint8_t *compressed_data;

        size_t max_bytes;

        ssize_t bytes;



        compressed_data = data;

        max_bytes = UBOOT_MAX_GUNZIP_BYTES;

        data = qemu_malloc(max_bytes);



        bytes = gunzip(data, max_bytes, compressed_data, hdr->ih_size);

        qemu_free(compressed_data);

        if (bytes < 0) {

            fprintf(stderr, "Unable to decompress gzipped image!\n");

            goto out;

        }

        hdr->ih_size = bytes;

    }



    cpu_physical_memory_write_rom(hdr->ih_load, data, hdr->ih_size);



    if (loadaddr)

        *loadaddr = hdr->ih_load;



    ret = hdr->ih_size;



out:

    if (data)

        qemu_free(data);

    close(fd);

    return ret;

}
