int qemu_uuid_parse(const char *str, uint8_t *uuid)

{

    int ret;



    if(strlen(str) != 36)

        return -1;



    ret = sscanf(str, UUID_FMT, &uuid[0], &uuid[1], &uuid[2], &uuid[3],

            &uuid[4], &uuid[5], &uuid[6], &uuid[7], &uuid[8], &uuid[9],

            &uuid[10], &uuid[11], &uuid[12], &uuid[13], &uuid[14], &uuid[15]);



    if(ret != 16)

        return -1;



#ifdef TARGET_I386

    smbios_add_field(1, offsetof(struct smbios_type_1, uuid), 16, uuid);

#endif



    return 0;

}
