static void bswap_shdr(struct elf_shdr *shdr)

{

    bswap32s(&shdr->sh_name);

    bswap32s(&shdr->sh_type);

    bswaptls(&shdr->sh_flags);

    bswaptls(&shdr->sh_addr);

    bswaptls(&shdr->sh_offset);

    bswaptls(&shdr->sh_size);

    bswap32s(&shdr->sh_link);

    bswap32s(&shdr->sh_info);

    bswaptls(&shdr->sh_addralign);

    bswaptls(&shdr->sh_entsize);

}
