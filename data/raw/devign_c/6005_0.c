void pmac_format_nvram_partition (MacIONVRAMState *nvr, int len)

{

    unsigned int i;

    uint32_t start = 0, end;

    struct OpenBIOS_nvpart_v1 *part_header;



    // OpenBIOS nvram variables

    // Variable partition

    part_header = (struct OpenBIOS_nvpart_v1 *)nvr->data;

    part_header->signature = OPENBIOS_PART_SYSTEM;

    pstrcpy(part_header->name, sizeof(part_header->name), "system");



    end = start + sizeof(struct OpenBIOS_nvpart_v1);

    for (i = 0; i < nb_prom_envs; i++)

        end = OpenBIOS_set_var(nvr->data, end, prom_envs[i]);



    // End marker

    nvr->data[end++] = '\0';



    end = start + ((end - start + 15) & ~15);

    /* XXX: OpenBIOS is not able to grow up a partition. Leave some space for

       new variables. */

    if (end < DEF_SYSTEM_SIZE)

        end = DEF_SYSTEM_SIZE;

    OpenBIOS_finish_partition(part_header, end - start);



    // free partition

    start = end;

    part_header = (struct OpenBIOS_nvpart_v1 *)&nvr->data[start];

    part_header->signature = OPENBIOS_PART_FREE;

    pstrcpy(part_header->name, sizeof(part_header->name), "free");



    end = len;

    OpenBIOS_finish_partition(part_header, end - start);

}
