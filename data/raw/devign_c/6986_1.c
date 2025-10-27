int load_image_targphys(const char *filename,

			target_phys_addr_t addr, int max_sz)

{

    int size;



    size = get_image_size(filename);

    if (size > 0)

        rom_add_file_fixed(filename, addr, -1);

    return size;

}
