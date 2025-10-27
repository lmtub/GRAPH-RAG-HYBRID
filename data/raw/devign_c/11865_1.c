ram_addr_t qemu_ram_addr_from_host_nofail(void *ptr)

{

    ram_addr_t ram_addr;



    if (qemu_ram_addr_from_host(ptr, &ram_addr)) {

        fprintf(stderr, "Bad ram pointer %p\n", ptr);

        abort();

    }

    return ram_addr;

}
